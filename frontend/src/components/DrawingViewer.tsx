import React, { useState, useEffect } from 'react';
import { Box, IconButton, Typography, Paper } from '@mui/material';
import { ChevronLeft, ChevronRight } from '@mui/icons-material';
import type { HoleAnnotation } from '../types';

interface DrawingViewerProps {
  sessionId: string | null;
  pageCount: number;
  annotations: HoleAnnotation[];
  selectedAnnotationId: string | null;
  onAnnotationSelect?: (annotationId: string) => void;
}

export const DrawingViewer: React.FC<DrawingViewerProps> = ({
  sessionId,
  pageCount,
  annotations,
  selectedAnnotationId,
  onAnnotationSelect,
}) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  useEffect(() => {
    if (sessionId && currentPage > 0) {
      const url = `http://localhost:8000/api/drawing/page/${currentPage}?session_id=${sessionId}`;
      setImageUrl(url);
    }
  }, [sessionId, currentPage]);

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < pageCount) {
      setCurrentPage(currentPage + 1);
    }
  };

  const pageAnnotations = annotations.filter(a => a.page === currentPage);

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Drawing Viewer</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton onClick={handlePrevPage} disabled={currentPage <= 1} size="small">
            <ChevronLeft />
          </IconButton>
          <Typography variant="body2">
            Page {currentPage} / {pageCount || 1}
          </Typography>
          <IconButton onClick={handleNextPage} disabled={currentPage >= pageCount} size="small">
            <ChevronRight />
          </IconButton>
        </Box>
      </Box>

      <Box sx={{ flex: 1, position: 'relative', overflow: 'auto', bgcolor: '#f5f5f5', borderRadius: 1 }}>
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={`Page ${currentPage}`}
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
            }}
          />
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <Typography color="text.secondary">No drawing loaded</Typography>
          </Box>
        )}
      </Box>

      {pageAnnotations.length > 0 && (
        <Box sx={{ mt: 2, maxHeight: 200, overflow: 'auto' }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Annotations on this page ({pageAnnotations.length}):
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {pageAnnotations.map((ann) => (
              <Paper
                key={ann.annotation_id}
                onClick={() => onAnnotationSelect?.(ann.annotation_id)}
                sx={{
                  p: 1,
                  cursor: 'pointer',
                  bgcolor: selectedAnnotationId === ann.annotation_id ? '#e3f2fd' : 'white',
                  border: selectedAnnotationId === ann.annotation_id ? '2px solid #2196f3' : '1px solid #e0e0e0',
                  '&:hover': {
                    bgcolor: '#f5f5f5',
                  },
                }}
              >
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {ann.annotation_id}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                  {ann.hole_type} {ann.diameter ? `∅${ann.diameter.toFixed(3)}` : ''}
                  {ann.count > 1 ? ` (${ann.count}X)` : ''}
                </Typography>
              </Paper>
            ))}
          </Box>
        </Box>
      )}
    </Paper>
  );
};
