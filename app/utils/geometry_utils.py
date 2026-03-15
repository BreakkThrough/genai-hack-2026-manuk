"""Geometry helper functions for 3D feature analysis."""

from __future__ import annotations

import math

import numpy as np

from app.models.schemas import Point3D, Vector3D


def distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def vectors_parallel(a: Vector3D, b: Vector3D, tol_deg: float = 5.0) -> bool:
    """Check if two direction vectors are parallel (or anti-parallel) within *tol_deg* degrees."""
    va = np.array([a.dx, a.dy, a.dz])
    vb = np.array([b.dx, b.dy, b.dz])
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return False
    cos_angle = abs(np.dot(va, vb) / (na * nb))
    cos_angle = min(cos_angle, 1.0)
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg <= tol_deg


def axes_coaxial(
    center_a: Point3D, axis_a: Vector3D,
    center_b: Point3D, axis_b: Vector3D,
    linear_tol: float = 0.5,
    angle_tol_deg: float = 5.0,
) -> bool:
    """
    Two cylinders are co-axial if their axes are parallel and the
    perpendicular distance between the axes is below *linear_tol*.
    """
    if not vectors_parallel(axis_a, axis_b, angle_tol_deg):
        return False

    va = np.array([axis_a.dx, axis_a.dy, axis_a.dz])
    va = va / (np.linalg.norm(va) + 1e-15)

    delta = np.array([
        center_b.x - center_a.x,
        center_b.y - center_a.y,
        center_b.z - center_a.z,
    ])
    perp = delta - np.dot(delta, va) * va
    return float(np.linalg.norm(perp)) < linear_tol


def diameter_matches(d1: float, d2: float, rel_tol: float = 0.05) -> bool:
    """Check if two diameters match within a relative tolerance (default 5 %)."""
    if d1 == 0 and d2 == 0:
        return True
    avg = (abs(d1) + abs(d2)) / 2.0
    if avg < 1e-12:
        return True
    return abs(d1 - d2) / avg <= rel_tol
