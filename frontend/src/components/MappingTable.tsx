import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Select,
  MenuItem,
  Button,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import { Save } from '@mui/icons-material';
import type { LinkageResult, FeatureMapping, HoleFeature3D } from '../types';

interface MappingTableProps {
  linkageResult: LinkageResult | null;
  onSave?: (mappings: Array<{ annotation_id: string; hole_ids: string[] }>) => void;
}

export const MappingTable: React.FC<MappingTableProps> = ({
  linkageResult,
  onSave,
}) => {
  const [editedMappings, setEditedMappings] = useState<Record<string, string[]>>({});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (linkageResult) {
      const initial: Record<string, string[]> = {};
      linkageResult.mappings.forEach((m) => {
        initial[m.annotation_id] = m.hole_ids;
      });
      setEditedMappings(initial);
      setHasChanges(false);
    }
  }, [linkageResult]);

  if (!linkageResult) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography color="text.secondary">
          No correlation results available. Complete Layer 4 first.
        </Typography>
      </Paper>
    );
  }

  const availableHoles = linkageResult.features_3d.holes;

  const handleMappingChange = (annotationId: string, event: SelectChangeEvent<string[]>) => {
    const newHoleIds = event.target.value as string[];
    setEditedMappings({
      ...editedMappings,
      [annotationId]: newHoleIds,
    });
    setHasChanges(true);
  };

  const handleSave = () => {
    if (onSave) {
      const mappingsArray = Object.entries(editedMappings).map(([annotation_id, hole_ids]) => ({
        annotation_id,
        hole_ids,
      }));
      onSave(mappingsArray);
      setHasChanges(false);
    }
  };

  const getConfidenceColor = (confidence: string): string => {
    switch (confidence) {
      case 'high':
        return '#16a34a';
      case 'medium':
        return '#d97706';
      case 'low':
        return '#dc2626';
      default:
        return '#64748b';
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Annotation-to-Feature Mappings</Typography>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSave}
          disabled={!hasChanges}
        >
          Save Changes
        </Button>
      </Box>

      <TableContainer sx={{ maxHeight: 500 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Annotation ID</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Diameter</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Count</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Matched Hole IDs</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Confidence</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Reasons</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {linkageResult.mappings.map((mapping: FeatureMapping) => {
              const annotation = linkageResult.annotations.annotations.find(
                (a) => a.annotation_id === mapping.annotation_id
              );

              return (
                <TableRow key={mapping.annotation_id} hover>
                  <TableCell>{mapping.annotation_id}</TableCell>
                  <TableCell>
                    {annotation?.diameter 
                      ? `∅${annotation.diameter.toFixed(3)} ${linkageResult.annotations.unit}`
                      : '-'}
                  </TableCell>
                  <TableCell>
                    <Chip label={annotation?.hole_type || '-'} size="small" />
                  </TableCell>
                  <TableCell>{annotation?.count || 1}</TableCell>
                  <TableCell>
                    <Select
                      multiple
                      value={editedMappings[mapping.annotation_id] || []}
                      onChange={(e) => handleMappingChange(mapping.annotation_id, e)}
                      size="small"
                      sx={{ minWidth: 150 }}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {(selected as string[]).map((value) => (
                            <Chip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {availableHoles.map((hole: HoleFeature3D) => (
                        <MenuItem key={hole.hole_id} value={hole.hole_id}>
                          {hole.hole_id} (∅{hole.primary_diameter.toFixed(4)})
                        </MenuItem>
                      ))}
                    </Select>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={mapping.confidence.toUpperCase()}
                      size="small"
                      sx={{
                        bgcolor: getConfidenceColor(mapping.confidence),
                        color: 'white',
                        fontWeight: 600,
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption">
                      {mapping.match_reasons.join('; ')}
                    </Typography>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};
