import React, { useState } from 'react';
import { Box, Typography, Paper, Button, TextField } from '@mui/material';
import { Download } from '@mui/icons-material';
import type { LinkageResult } from '../types';

interface JSONExportPanelProps {
  linkageResult: LinkageResult | null;
  sessionId: string | null;
}

export const JSONExportPanel: React.FC<JSONExportPanelProps> = ({
  linkageResult,
  sessionId,
}) => {
  const [expanded, setExpanded] = useState(false);

  if (!linkageResult) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography color="text.secondary">
          No results available for export. Complete the pipeline first.
        </Typography>
      </Paper>
    );
  }

  const jsonString = JSON.stringify(linkageResult, null, 2);
  const previewLength = 2000;
  const preview = expanded ? jsonString : jsonString.slice(0, previewLength);

  const handleDownload = () => {
    if (!sessionId) return;

    const url = `http://localhost:8000/api/export?session_id=${sessionId}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = 'linkage_result.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(jsonString);
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">JSON Export</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            onClick={handleCopyToClipboard}
          >
            Copy to Clipboard
          </Button>
          <Button
            variant="contained"
            startIcon={<Download />}
            onClick={handleDownload}
          >
            Download JSON
          </Button>
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Complete linkage result with all annotations, 3D features, and mappings:
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Paper sx={{ p: 1.5, flex: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {linkageResult.annotations.annotations.length}
            </Typography>
            <Typography variant="caption" color="text.secondary">Annotations</Typography>
          </Paper>
          <Paper sx={{ p: 1.5, flex: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {linkageResult.features_3d.holes.length}
            </Typography>
            <Typography variant="caption" color="text.secondary">3D Holes</Typography>
          </Paper>
          <Paper sx={{ p: 1.5, flex: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {linkageResult.mappings.length}
            </Typography>
            <Typography variant="caption" color="text.secondary">Mappings</Typography>
          </Paper>
        </Box>
      </Box>

      <TextField
        multiline
        fullWidth
        value={preview}
        InputProps={{
          readOnly: true,
          sx: {
            fontFamily: 'monospace',
            fontSize: '0.875rem',
          },
        }}
        minRows={10}
        maxRows={20}
      />

      {!expanded && jsonString.length > previewLength && (
        <Button
          size="small"
          onClick={() => setExpanded(true)}
          sx={{ mt: 1 }}
        >
          Show Full JSON ({jsonString.length} characters)
        </Button>
      )}
    </Paper>
  );
};
