import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  CircularProgress,
} from '@mui/material';
import { Upload, PlayArrow } from '@mui/icons-material';
import type { PipelineStep, StepStatus } from '../types';

interface SidebarProps {
  sessionId: string | null;
  currentStep: PipelineStep;
  stepStatuses: Record<PipelineStep, StepStatus>;
  onUpload: (pdfFile: File | null, stepFile: File | null, unit: string) => void;
  onRunNextLayer: () => void;
  isLoading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessionId,
  currentStep,
  stepStatuses,
  onUpload,
  onRunNextLayer,
  isLoading,
}) => {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [stepFile, setStepFile] = useState<File | null>(null);
  const [unit, setUnit] = useState<string>('inch');

  const handlePdfChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setPdfFile(event.target.files[0]);
    }
  };

  const handleStepChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setStepFile(event.target.files[0]);
    }
  };

  const handleUpload = () => {
    onUpload(pdfFile, stepFile, unit);
  };

  const canRunNextLayer = (): boolean => {
    if (!sessionId || isLoading) return false;
    
    if (currentStep === 1 && stepStatuses[1] !== 'completed') return true;
    if (currentStep === 2 && stepStatuses[2] !== 'completed' && stepStatuses[1] === 'completed') return true;
    if (currentStep === 3 && stepStatuses[3] !== 'completed' && stepStatuses[2] === 'completed') return true;
    if (currentStep === 4 && stepStatuses[4] !== 'completed' && stepStatuses[2] === 'completed' && stepStatuses[3] === 'completed') return true;
    
    return false;
  };

  const getNextLayerLabel = (): string => {
    if (stepStatuses[1] !== 'completed') return 'Run Layer 1 (OCR)';
    if (stepStatuses[2] !== 'completed') return 'Run Layer 2 (Vision)';
    if (stepStatuses[3] !== 'completed') return 'Run Layer 3 (STEP)';
    if (stepStatuses[4] !== 'completed') return 'Run Layer 4 (Correlation)';
    return 'Pipeline Complete';
  };

  return (
    <Paper
      sx={{
        width: 280,
        height: '100vh',
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        overflow: 'auto',
      }}
    >
      <Typography variant="h6" sx={{ fontWeight: 600 }}>
        Drawing - 3D Hole Linker
      </Typography>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Upload Files
        </Typography>

        <Button
          variant="outlined"
          component="label"
          fullWidth
          size="small"
          sx={{ mb: 1, justifyContent: 'flex-start' }}
        >
          {pdfFile ? pdfFile.name : 'Select PDF Drawing'}
          <input
            type="file"
            hidden
            accept=".pdf"
            onChange={handlePdfChange}
          />
        </Button>

        <Button
          variant="outlined"
          component="label"
          fullWidth
          size="small"
          sx={{ mb: 1, justifyContent: 'flex-start' }}
        >
          {stepFile ? stepFile.name : 'Select STEP File'}
          <input
            type="file"
            hidden
            accept=".stp,.step"
            onChange={handleStepChange}
          />
        </Button>

        <FormControl fullWidth size="small" sx={{ mb: 1 }}>
          <InputLabel>Units</InputLabel>
          <Select
            value={unit}
            label="Units"
            onChange={(e) => setUnit(e.target.value)}
          >
            <MenuItem value="inch">Inch</MenuItem>
            <MenuItem value="mm">Millimeter</MenuItem>
          </Select>
        </FormControl>

        <Button
          variant="contained"
          fullWidth
          startIcon={<Upload />}
          onClick={handleUpload}
          disabled={!pdfFile || !stepFile || isLoading}
        >
          Upload & Initialize
        </Button>
      </Box>

      <Divider />

      {sessionId && (
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Pipeline Control
          </Typography>

          <Button
            variant="contained"
            fullWidth
            startIcon={isLoading ? <CircularProgress size={16} /> : <PlayArrow />}
            onClick={onRunNextLayer}
            disabled={!canRunNextLayer()}
            color="primary"
          >
            {isLoading ? 'Running...' : getNextLayerLabel()}
          </Button>

          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Session ID:
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {sessionId.slice(0, 8)}...
            </Typography>
          </Box>
        </Box>
      )}

      <Box sx={{ mt: 'auto', pt: 2 }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Vision Model: GPT-4o
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          Units: {unit}
        </Typography>
      </Box>
    </Paper>
  );
};
