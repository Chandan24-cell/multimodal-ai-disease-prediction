// frontend/src/charts/ConfidenceChart.jsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function ConfidenceChart({ predictions }) {
  const data = Object.entries(predictions)
    .map(([name, confidence]) => ({ name, confidence: (confidence * 100).toFixed(1) }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 8); // Top 8 predictions

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 100]} unit="%" />
        <YAxis dataKey="name" type="category" width={120} />
        <Tooltip formatter={(value) => `${value}%`} />
        <Bar dataKey="confidence" fill="#1976d2" />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default ConfidenceChart;