// frontend/src/charts/HeatmapViewer.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

function HeatmapViewer({ heatmapBase64 }) {
  if (!heatmapBase64) return <Typography color="text.secondary">No heatmap available</Typography>;

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2" gutterBottom>Grad-CAM Heatmap</Typography>
      <img src={heatmapBase64} alt="Grad-CAM Heatmap" style={{ width: '100%', borderRadius: 8 }} />
    </Box>
  );
}

export default HeatmapViewer;