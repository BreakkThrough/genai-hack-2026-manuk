import React from 'react';
import { Box, Typography, Paper, Chip, Divider } from '@mui/material';
import type {
  PipelineStep,
  Layer1Response,
  Layer2Response,
  Layer3Response,
  Layer4Response,
  HoleAnnotation,
  HoleFeature3D,
  FeatureMapping,
} from '../types';

interface LayerOutputPanelProps {
  currentStep: PipelineStep;
  layer1Data: Layer1Response | null;
  layer2Data: Layer2Response | null;
  layer3Data: Layer3Response | null;
  layer4Data: Layer4Response | null;
  onAnnotationClick?: (annotationId: string) => void;
}

export const LayerOutputPanel: React.FC<LayerOutputPanelProps> = ({
  currentStep,
  layer1Data,
  layer2Data,
  layer3Data,
  layer4Data,
  onAnnotationClick,
}) => {
  const renderLayer1 = () => {
    if (!layer1Data) {
      return <Typography color="text.secondary">Layer 1 not started</Typography>;
    }

    if (layer1Data.error) {
      return (
        <Box>
          <Typography color="error" sx={{ mb: 1 }}>Error: {layer1Data.error}</Typography>
          <Typography variant="body2" color="text.secondary">
            Pipeline will continue in vision-only mode.
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Paper sx={{ p: 2, flex: 1 }}>
            <Typography variant="h4" color="primary">{layer1Data.page_count}</Typography>
            <Typography variant="caption" color="text.secondary">Pages Processed</Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1 }}>
            <Typography variant="h4" color="primary">{layer1Data.element_count}</Typography>
            <Typography variant="caption" color="text.secondary">Text Elements</Typography>
          </Paper>
        </Box>

        {layer1Data.sample_elements.length > 0 && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Sample OCR Elements:</Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {layer1Data.sample_elements.slice(0, 10).map((elem, idx) => (
                <Paper key={idx} sx={{ p: 1, mb: 1, bgcolor: '#f8fafc' }}>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {elem.text}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Confidence: {(elem.confidence * 100).toFixed(1)}%
                  </Typography>
                </Paper>
              ))}
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  const renderLayer2 = () => {
    if (!layer2Data) {
      return <Typography color="text.secondary">Layer 2 not started</Typography>;
    }

    if (layer2Data.error) {
      return <Typography color="error">Error: {layer2Data.error}</Typography>;
    }

    if (!layer2Data.annotations) {
      return <Typography color="text.secondary">No annotations extracted</Typography>;
    }

    const annotations = layer2Data.annotations.annotations;

    return (
      <Box>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Paper sx={{ p: 2, flex: 1 }}>
            <Typography variant="h4" color="primary">{annotations.length}</Typography>
            <Typography variant="caption" color="text.secondary">Hole Annotations</Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1 }}>
            <Typography variant="h6" color="primary">{layer2Data.annotations.unit}</Typography>
            <Typography variant="caption" color="text.secondary">Drawing Units</Typography>
          </Paper>
        </Box>

        <Typography variant="subtitle2" sx={{ mb: 1 }}>Annotations:</Typography>
        <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
          {annotations.map((ann: HoleAnnotation) => (
            <Paper
              key={ann.annotation_id}
              onClick={() => onAnnotationClick?.(ann.annotation_id)}
              sx={{
                p: 1.5,
                mb: 1,
                cursor: 'pointer',
                '&:hover': { bgcolor: '#f5f5f5' },
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {ann.annotation_id}
                </Typography>
                <Chip label={ann.hole_type} size="small" />
              </Box>
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                {ann.diameter ? `∅${ann.diameter.toFixed(3)} ${layer2Data.annotations.unit}` : 'No diameter'}
                {ann.count > 1 && ` (${ann.count}X)`}
              </Typography>
              {ann.thread_spec && (
                <Typography variant="caption" color="text.secondary">
                  Thread: {ann.thread_spec.designation}
                </Typography>
              )}
              {ann.raw_text && (
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontStyle: 'italic' }}>
                  "{ann.raw_text}"
                </Typography>
              )}
            </Paper>
          ))}
        </Box>
      </Box>
    );
  };

  const renderLayer3 = () => {
    if (!layer3Data) {
      return <Typography color="text.secondary">Layer 3 not started</Typography>;
    }

    if (layer3Data.error) {
      return <Typography color="error">Error: {layer3Data.error}</Typography>;
    }

    if (!layer3Data.step_features) {
      return <Typography color="text.secondary">No STEP features parsed</Typography>;
    }

    const holes = layer3Data.step_features.holes;
    const uniqueDiameters = Array.from(new Set(holes.map(h => h.primary_diameter))).sort((a, b) => a - b);

    return (
      <Box>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Paper sx={{ p: 2, flex: 1 }}>
            <Typography variant="h4" color="primary">{layer3Data.step_features.total_cylindrical_faces}</Typography>
            <Typography variant="caption" color="text.secondary">Cylindrical Faces</Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1 }}>
            <Typography variant="h4" color="primary">{holes.length}</Typography>
            <Typography variant="caption" color="text.secondary">Grouped Holes</Typography>
          </Paper>
        </Box>

        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Unique Diameters: {uniqueDiameters.map(d => d.toFixed(3)).join(', ')}
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" sx={{ mb: 1 }}>3D Hole Features:</Typography>
        <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
          {holes.map((hole: HoleFeature3D) => (
            <Paper key={hole.hole_id} sx={{ p: 1.5, mb: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {hole.hole_id}
                </Typography>
                <Chip label={hole.hole_type} size="small" />
              </Box>
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                ∅{hole.primary_diameter.toFixed(4)}
                {hole.counterbore_diameter && ` | CB ∅${hole.counterbore_diameter.toFixed(4)}`}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Center: ({hole.center.x.toFixed(2)}, {hole.center.y.toFixed(2)}, {hole.center.z.toFixed(2)})
              </Typography>
            </Paper>
          ))}
        </Box>
      </Box>
    );
  };

  const renderLayer4 = () => {
    if (!layer4Data) {
      return <Typography color="text.secondary">Layer 4 not started</Typography>;
    }

    if (layer4Data.error) {
      return <Typography color="error">Error: {layer4Data.error}</Typography>;
    }

    if (!layer4Data.linkage_result) {
      return <Typography color="text.secondary">No correlation results</Typography>;
    }

    const linkage = layer4Data.linkage_result;
    const highCount = linkage.mappings.filter(m => m.confidence === 'high').length;
    const mediumCount = linkage.mappings.filter(m => m.confidence === 'medium').length;
    const lowCount = linkage.mappings.filter(m => m.confidence === 'low').length;

    return (
      <Box>
        <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
          <Paper sx={{ p: 2, flex: 1, minWidth: 120 }}>
            <Typography variant="h4" color="primary">{linkage.mappings.length}</Typography>
            <Typography variant="caption" color="text.secondary">Mappings</Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 120 }}>
            <Typography variant="h4" sx={{ color: '#16a34a' }}>{highCount}</Typography>
            <Typography variant="caption" color="text.secondary">High Confidence</Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 120 }}>
            <Typography variant="h4" sx={{ color: '#d97706' }}>{mediumCount}</Typography>
            <Typography variant="caption" color="text.secondary">Medium</Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 120 }}>
            <Typography variant="h4" sx={{ color: '#dc2626' }}>{lowCount}</Typography>
            <Typography variant="caption" color="text.secondary">Low</Typography>
          </Paper>
        </Box>

        {linkage.unmapped_annotations.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="warning.main">
              Unmapped annotations: {linkage.unmapped_annotations.join(', ')}
            </Typography>
          </Box>
        )}

        {linkage.unmapped_holes.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="warning.main">
              Unmapped 3D holes: {linkage.unmapped_holes.join(', ')}
            </Typography>
          </Box>
        )}

        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          See Mapping Table below for detailed view and editing.
        </Typography>
      </Box>
    );
  };

  return (
    <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Layer {currentStep} Output
      </Typography>
      
      {currentStep === 1 && renderLayer1()}
      {currentStep === 2 && renderLayer2()}
      {currentStep === 3 && renderLayer3()}
      {currentStep === 4 && renderLayer4()}
    </Paper>
  );
};
