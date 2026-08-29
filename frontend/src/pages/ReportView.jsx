// frontend/src/pages/ReportView.jsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Container, Paper, Typography, Box, CircularProgress, Alert } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import { reportService } from '../services/api';

function ReportView() {
  const { reportId } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const data = await reportService.getReport(reportId);
        setReport(data);
      } catch (err) {
        setError('Failed to load report: ' + err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
  }, [reportId]);

  if (loading) return <CircularProgress sx={{ display: 'block', margin: '50px auto' }} />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!report) return null;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom color="primary">
          AI-Generated Clinical Report
        </Typography>
        
        <Box sx={{ mb: 3, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
          <Typography variant="body2">{report.disclaimer}</Typography>
        </Box>

        <Box sx={{ 
          '& h2': { color: 'primary.main', mt: 3, mb: 2 },
          '& p': { mb: 2 },
          '& strong': { color: 'text.primary' }
        }}>
          <ReactMarkdown>{report.generated_report}</ReactMarkdown>
        </Box>

        {report.retrieved_references.length > 0 && (
          <Box sx={{ mt: 4, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
            <Typography variant="h6" gutterBottom>📚 Retrieved References</Typography>
            <ul>
              {report.retrieved_references.map((ref, idx) => (
                <li key={idx}>{ref}</li>
              ))}
            </ul>
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default ReportView;