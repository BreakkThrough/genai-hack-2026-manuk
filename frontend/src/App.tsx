import React, { useState } from 'react';
import { Box, Container, Grid, Paper, Typography, Alert } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import { Sidebar } from './components/Sidebar';
import { PipelineStepBar } from './components/PipelineStepBar';
import { DrawingViewer } from './components/DrawingViewer';
import { LayerOutputPanel } from './components/LayerOutputPanel';
import { MappingTable } from './components/MappingTable';
import { JSONExportPanel } from './components/JSONExportPanel';
import { apiClient } from './api';

import type {
  PipelineStep,
  StepStatus,
  Layer1Response,
  Layer2Response,
  Layer3Response,
  Layer4Response,
  HoleAnnotation,
} from './types';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2563eb',
    },
    secondary: {
      main: '#16a34a',
    },
  },
});

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<PipelineStep>(1);
  const [stepStatuses, setStepStatuses] = useState<Record<PipelineStep, StepStatus>>({
    1: 'pending',
    2: 'pending',
    3: 'pending',
    4: 'pending',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);

  const [layer1Data, setLayer1Data] = useState<Layer1Response | null>(null);
  const [layer2Data, setLayer2Data] = useState<Layer2Response | null>(null);
  const [layer3Data, setLayer3Data] = useState<Layer3Response | null>(null);
  const [layer4Data, setLayer4Data] = useState<Layer4Response | null>(null);

  const handleUpload = async (pdfFile: File | null, stepFile: File | null, unit: string) => {
    if (!pdfFile || !stepFile) {
      setError('Please select both PDF and STEP files');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.uploadFiles(pdfFile, stepFile, unit);
      setSessionId(response.session_id);
      
      if (response.step_features) {
        setLayer3Data({
          status: 'success',
          step_features: response.step_features,
          hole_count: response.step_features.holes.length,
          error: null,
        });
        setStepStatuses(prev => ({ ...prev, 3: 'completed' }));
      }

      setCurrentStep(1);
      setStepStatuses({
        1: 'pending',
        2: 'pending',
        3: response.step_features ? 'completed' : 'pending',
        4: 'pending',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunNextLayer = async () => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      if (stepStatuses[1] !== 'completed') {
        setStepStatuses(prev => ({ ...prev, 1: 'running' }));
        setCurrentStep(1);
        const result = await apiClient.runLayer1(sessionId);
        setLayer1Data(result);
        
        if (result.error) {
          setStepStatuses(prev => ({ ...prev, 1: 'error' }));
          setError(`Layer 1: ${result.error}`);
        } else {
          setStepStatuses(prev => ({ ...prev, 1: 'completed' }));
          setPageCount(result.page_count);
        }
      } else if (stepStatuses[2] !== 'completed') {
        setStepStatuses(prev => ({ ...prev, 2: 'running' }));
        setCurrentStep(2);
        const result = await apiClient.runLayer2(sessionId);
        setLayer2Data(result);
        
        if (result.error) {
          setStepStatuses(prev => ({ ...prev, 2: 'error' }));
          setError(`Layer 2: ${result.error}`);
        } else {
          setStepStatuses(prev => ({ ...prev, 2: 'completed' }));
        }
      } else if (stepStatuses[3] !== 'completed') {
        setStepStatuses(prev => ({ ...prev, 3: 'running' }));
        setCurrentStep(3);
        const result = await apiClient.runLayer3(sessionId);
        setLayer3Data(result);
        
        if (result.error) {
          setStepStatuses(prev => ({ ...prev, 3: 'error' }));
          setError(`Layer 3: ${result.error}`);
        } else {
          setStepStatuses(prev => ({ ...prev, 3: 'completed' }));
        }
      } else if (stepStatuses[4] !== 'completed') {
        setStepStatuses(prev => ({ ...prev, 4: 'running' }));
        setCurrentStep(4);
        const result = await apiClient.runLayer4(sessionId);
        setLayer4Data(result);
        
        if (result.error) {
          setStepStatuses(prev => ({ ...prev, 4: 'error' }));
          setError(`Layer 4: ${result.error}`);
        } else {
          setStepStatuses(prev => ({ ...prev, 4: 'completed' }));
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Pipeline step failed');
      setStepStatuses(prev => ({ ...prev, [currentStep]: 'error' }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveMappings = async (mappings: Array<{ annotation_id: string; hole_ids: string[] }>) => {
    if (!sessionId) return;

    try {
      await apiClient.updateMappings(sessionId, mappings);
      
      if (layer4Data?.linkage_result) {
        const updatedLinkage = { ...layer4Data.linkage_result };
        mappings.forEach(({ annotation_id, hole_ids }) => {
          const mapping = updatedLinkage.mappings.find(m => m.annotation_id === annotation_id);
          if (mapping) {
            mapping.hole_ids = hole_ids;
          }
        });
        setLayer4Data({ ...layer4Data, linkage_result: updatedLinkage });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save mappings');
    }
  };

  const allAnnotations: HoleAnnotation[] = layer2Data?.annotations?.annotations || [];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <Sidebar
          sessionId={sessionId}
          currentStep={currentStep}
          stepStatuses={stepStatuses}
          onUpload={handleUpload}
          onRunNextLayer={handleRunNextLayer}
          isLoading={isLoading}
        />

        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Box sx={{ p: 3, bgcolor: 'white', borderBottom: '1px solid #e0e0e0' }}>
            <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>
              GenAI-Powered Drawing - 3D Hole Feature Linker
            </Typography>
            <PipelineStepBar
              currentStep={currentStep}
              stepStatuses={stepStatuses}
              onStepClick={setCurrentStep}
            />
            {error && (
              <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}
          </Box>

          <Box sx={{ flex: 1, overflow: 'auto', p: 3, bgcolor: '#f5f5f5' }}>
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={7}>
                <DrawingViewer
                  sessionId={sessionId}
                  pageCount={pageCount}
                  annotations={allAnnotations}
                  selectedAnnotationId={selectedAnnotationId}
                  onAnnotationSelect={setSelectedAnnotationId}
                />
              </Grid>
              <Grid item xs={12} md={5}>
                <LayerOutputPanel
                  currentStep={currentStep}
                  layer1Data={layer1Data}
                  layer2Data={layer2Data}
                  layer3Data={layer3Data}
                  layer4Data={layer4Data}
                  onAnnotationClick={setSelectedAnnotationId}
                />
              </Grid>
            </Grid>

            {stepStatuses[4] === 'completed' && layer4Data?.linkage_result && (
              <>
                <Box sx={{ mb: 3 }}>
                  <MappingTable
                    linkageResult={layer4Data.linkage_result}
                    onSave={handleSaveMappings}
                  />
                </Box>

                <Box>
                  <JSONExportPanel
                    linkageResult={layer4Data.linkage_result}
                    sessionId={sessionId}
                  />
                </Box>
              </>
            )}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
