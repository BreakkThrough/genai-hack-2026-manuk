"""
Extract cylindrical hole features from STEP (ISO 10303-21) files.

Uses pure-Python text parsing of the STEP physical file format -- no compiled
CAD kernel (pythonocc / OCC) required.  The parser resolves the entity
reference chain:

    CYLINDRICAL_SURFACE  ->  AXIS2_PLACEMENT_3D  ->  CARTESIAN_POINT + DIRECTION

to recover each cylinder's radius, centre point and axis direction, then
groups co-axial cylinders into logical hole features.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from pathlib import Path

from app.models.schemas import (
    CylindricalFeature,
    HoleFeature3D,
    HoleType,
    Point3D,
    StepFeatures,
    Vector3D,
)
from app.utils.geometry_utils import axes_coaxial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level STEP entity parsing
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(
    r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\(([^;]*)\)\s*;",
    re.DOTALL,
)

_TUPLE_RE = re.compile(r"\(\s*([^)]+)\)")

_REF_RE = re.compile(r"#(\d+)")


def _parse_step_entities(text: str) -> dict[int, tuple[str, str]]:
    """Return {entity_id: (entity_type, raw_args_string)} for every entity."""
    entities: dict[int, tuple[str, str]] = {}
    for m in _ENTITY_RE.finditer(text):
        eid = int(m.group(1))
        etype = m.group(2)
        args = m.group(3).strip()
        entities[eid] = (etype, args)
    return entities


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _resolve_cartesian_point(
    entities: dict[int, tuple[str, str]], ref_id: int
) -> Point3D | None:
    """Resolve a CARTESIAN_POINT entity to a Point3D."""
    entry = entities.get(ref_id)
    if entry is None:
        return None
    etype, args = entry
    if etype != "CARTESIAN_POINT":
        return None
    m = _TUPLE_RE.search(args)
    if not m:
        return None
    coords = [float(v.strip()) for v in m.group(1).split(",") if _is_float(v.strip())]
    if len(coords) < 3:
        return None
    return Point3D(x=coords[0], y=coords[1], z=coords[2])


def _resolve_direction(
    entities: dict[int, tuple[str, str]], ref_id: int
) -> Vector3D | None:
    """Resolve a DIRECTION entity to a Vector3D."""
    entry = entities.get(ref_id)
    if entry is None:
        return None
    etype, args = entry
    if etype != "DIRECTION":
        return None
    m = _TUPLE_RE.search(args)
    if not m:
        return None
    coords = [float(v.strip()) for v in m.group(1).split(",") if _is_float(v.strip())]
    if len(coords) < 3:
        return None
    return Vector3D(dx=coords[0], dy=coords[1], dz=coords[2])


def _resolve_axis2_placement(
    entities: dict[int, tuple[str, str]], ref_id: int
) -> tuple[Point3D | None, Vector3D | None]:
    """Resolve AXIS2_PLACEMENT_3D -> (location, axis_direction)."""
    entry = entities.get(ref_id)
    if entry is None:
        return None, None
    etype, args = entry
    if etype != "AXIS2_PLACEMENT_3D":
        return None, None
    refs = _REF_RE.findall(args)
    if len(refs) < 2:
        return None, None
    point_id = int(refs[0])
    dir_id = int(refs[1])
    return (
        _resolve_cartesian_point(entities, point_id),
        _resolve_direction(entities, dir_id),
    )


# ---------------------------------------------------------------------------
# Cylinder extraction
# ---------------------------------------------------------------------------

def _extract_cylinders(
    entities: dict[int, tuple[str, str]],
) -> list[CylindricalFeature]:
    """Find all CYLINDRICAL_SURFACE entities and resolve their geometry."""
    features: list[CylindricalFeature] = []
    idx = 0

    for eid, (etype, args) in entities.items():
        if etype != "CYLINDRICAL_SURFACE":
            continue

        refs = _REF_RE.findall(args)
        if not refs:
            continue

        tokens = [t.strip().rstrip(")").lstrip("(") for t in args.split(",")]
        radius: float | None = None
        for tok in reversed(tokens):
            tok = tok.strip().rstrip(";").rstrip(")")
            if _is_float(tok) and not tok.startswith("#"):
                radius = float(tok)
                break

        if radius is None or radius <= 0:
            continue

        axis_ref = int(refs[0])
        center, axis_dir = _resolve_axis2_placement(entities, axis_ref)

        if center is None:
            center = Point3D(x=0, y=0, z=0)
        if axis_dir is None:
            axis_dir = Vector3D(dx=0, dy=0, dz=1)

        features.append(CylindricalFeature(
            feature_id=f"cyl_{idx}",
            diameter=round(radius * 2, 6),
            radius=round(radius, 6),
            depth=None,
            center=center,
            axis=axis_dir,
            is_through=False,
            face_index=eid,
        ))
        idx += 1

    logger.info("Parsed %d CYLINDRICAL_SURFACE entities", len(features))
    return features


# ---------------------------------------------------------------------------
# Depth estimation via CIRCLE entity analysis
# ---------------------------------------------------------------------------

def _estimate_depths(
    entities: dict[int, tuple[str, str]],
    cylinders: list[CylindricalFeature],
) -> None:
    """
    Estimate cylinder depth by finding CIRCLE entities that share the same
    axis placement.  Two circles on the same axis with matching radius give
    the depth (distance along the axis between the two planes).
    """
    circle_positions: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for eid, (etype, args) in entities.items():
        if etype != "CIRCLE":
            continue
        refs = _REF_RE.findall(args)
        tokens = [t.strip().rstrip(")").lstrip("(") for t in args.split(",")]
        radius = None
        for tok in reversed(tokens):
            tok = tok.strip().rstrip(";").rstrip(")")
            if _is_float(tok) and not tok.startswith("#"):
                radius = float(tok)
                break
        if radius is None or not refs:
            continue
        axis_ref = int(refs[0])
        center, _ = _resolve_axis2_placement(entities, axis_ref)
        if center:
            circle_positions[axis_ref].append((center.x, center.y, center.z, radius))

    for cyl in cylinders:
        entry = entities.get(cyl.face_index)
        if entry is None:
            continue
        _, args = entry
        refs = _REF_RE.findall(args)
        if not refs:
            continue
        axis_ref = int(refs[0])

        positions = circle_positions.get(axis_ref, [])
        matching = [p for p in positions if abs(p[3] - cyl.radius) < 0.001]
        if len(matching) >= 2:
            ax = [cyl.axis.dx, cyl.axis.dy, cyl.axis.dz]
            norm = math.sqrt(sum(a ** 2 for a in ax))
            if norm > 1e-12:
                ax = [a / norm for a in ax]
            projections = [
                m[0] * ax[0] + m[1] * ax[1] + m[2] * ax[2] for m in matching
            ]
            depth = max(projections) - min(projections)
            if depth > 1e-6:
                cyl.depth = round(depth, 6)


# ---------------------------------------------------------------------------
# Grouping into logical holes
# ---------------------------------------------------------------------------

def _group_into_holes(
    cylinders: list[CylindricalFeature],
    coaxial_tol: float = 0.5,
) -> list[HoleFeature3D]:
    """
    Group co-axial cylindrical surfaces into logical hole features.

    Two cylinders are grouped when their axes are parallel and nearly
    coincident.  Within each group the smallest diameter is the primary hole;
    a larger co-axial cylinder is treated as a counterbore.
    """
    if not cylinders:
        return []

    assigned: set[int] = set()
    groups: list[list[CylindricalFeature]] = []

    for i, c in enumerate(cylinders):
        if i in assigned:
            continue
        group = [c]
        assigned.add(i)
        for j in range(i + 1, len(cylinders)):
            if j in assigned:
                continue
            if axes_coaxial(
                c.center, c.axis,
                cylinders[j].center, cylinders[j].axis,
                coaxial_tol,
            ):
                group.append(cylinders[j])
                assigned.add(j)
        groups.append(group)

    holes: list[HoleFeature3D] = []
    for gidx, group in enumerate(groups):
        group_sorted = sorted(group, key=lambda f: f.diameter)
        primary = group_sorted[0]

        hole_type = HoleType.SIMPLE
        cb_dia: float | None = None
        cb_depth: float | None = None

        if len(group_sorted) > 1:
            larger = group_sorted[-1]
            if larger.diameter > primary.diameter * 1.2:
                hole_type = HoleType.COUNTERBORE
                cb_dia = larger.diameter
                cb_depth = larger.depth

        holes.append(HoleFeature3D(
            hole_id=f"hole_{gidx}",
            hole_type=hole_type,
            primary_diameter=primary.diameter,
            primary_depth=primary.depth,
            counterbore_diameter=cb_dia,
            counterbore_depth=cb_depth,
            center=primary.center,
            axis=primary.axis,
            is_through=False,
            cylinders=group,
        ))

    logger.info("Grouped %d cylinders into %d hole features", len(cylinders), len(holes))
    return holes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_step(step_path: str | Path) -> StepFeatures:
    """
    Full STEP parsing pipeline:
    1. Parse all STEP entities from the text file
    2. Extract CYLINDRICAL_SURFACE entities with resolved geometry
    3. Attempt depth estimation from CIRCLE entities
    4. Group co-axial cylinders into logical holes
    """
    step_path = Path(step_path)
    text = step_path.read_text(errors="replace")

    entities = _parse_step_entities(text)
    logger.info("Parsed %d total STEP entities from %s", len(entities), step_path.name)

    cylinders = _extract_cylinders(entities)
    _estimate_depths(entities, cylinders)
    holes = _group_into_holes(cylinders)

    return StepFeatures(
        source_step=str(step_path),
        total_cylindrical_faces=len(cylinders),
        holes=holes,
    )
