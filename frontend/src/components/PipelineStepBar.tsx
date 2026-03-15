import React from 'react';
import { Box, Stepper, Step, StepLabel, StepButton } from '@mui/material';
import type { PipelineStep, StepStatus } from '../types';

interface PipelineStepBarProps {
  currentStep: PipelineStep;
  stepStatuses: Record<PipelineStep, StepStatus>;
  onStepClick?: (step: PipelineStep) => void;
}

const STEP_LABELS: Record<PipelineStep, string> = {
  1: '1. OCR (Azure DI)',
  2: '2. Vision (GPT-4o)',
  3: '3. STEP Parse',
  4: '4. Correlation',
};

export const PipelineStepBar: React.FC<PipelineStepBarProps> = ({
  currentStep,
  stepStatuses,
  onStepClick,
}) => {
  const getStepColor = (step: PipelineStep): string => {
    const status = stepStatuses[step];
    switch (status) {
      case 'completed':
        return '#16a34a';
      case 'running':
        return '#2563eb';
      case 'error':
        return '#dc2626';
      default:
        return '#94a3b8';
    }
  };

  const getActiveStep = (): number => {
    if (stepStatuses[4] === 'completed') return 4;
    if (stepStatuses[3] === 'completed') return 3;
    if (stepStatuses[2] === 'completed') return 2;
    if (stepStatuses[1] === 'completed') return 1;
    return 0;
  };

  return (
    <Box sx={{ width: '100%', mb: 3 }}>
      <Stepper activeStep={getActiveStep()} alternativeLabel>
        {([1, 2, 3, 4] as PipelineStep[]).map((step) => (
          <Step key={step} completed={stepStatuses[step] === 'completed'}>
            <StepButton
              onClick={() => onStepClick?.(step)}
              sx={{
                '& .MuiStepLabel-label': {
                  color: getStepColor(step),
                  fontWeight: stepStatuses[step] === 'running' ? 600 : 400,
                },
              }}
            >
              <StepLabel
                error={stepStatuses[step] === 'error'}
                sx={{
                  '& .MuiStepIcon-root': {
                    color: getStepColor(step),
                  },
                }}
              >
                {STEP_LABELS[step]}
              </StepLabel>
            </StepButton>
          </Step>
        ))}
      </Stepper>
    </Box>
  );
};
