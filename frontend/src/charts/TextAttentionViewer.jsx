// frontend/src/charts/TextAttentionViewer.jsx
import React from 'react';
import { Box, Typography, Chip } from '@mui/material';

function TextAttentionViewer({ tokens }) {
  if (!tokens || tokens.length === 0) return <Typography color="text.secondary">No attention data available</Typography>;

  // Get top 10 most important tokens
  const topTokens = tokens.sort((a, b) => b.attention_weight - a.attention_weight).slice(0, 10);

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2" gutterBottom>Important Words (Attention Weights)</Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {topTokens.map((token, idx) => (
          <Chip
            key={idx}
            label={`${token.token} (${(token.attention_weight * 100).toFixed(0)}%)`}
            size="small"
            sx={{
              bgcolor: `rgba(25, 118, 210, ${token.attention_weight})`,
              color: token.attention_weight > 0.5 ? 'white' : 'black',
            }}
          />
        ))}
      </Box>
    </Box>
  );
}

export default TextAttentionViewer;