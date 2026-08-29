// frontend/src/pages/Dashboard.jsx
import React from 'react';
import {
  Box, AppBar, Toolbar, Typography, Button, Container, Grid, Card, CardContent, CardActions
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { Assessment, Upload, Description } from '@mui/icons-material';

function Dashboard({ onLogout }) {
  const navigate = useNavigate();

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🏥 Multimodal Healthcare AI
          </Typography>
          <Button color="inherit" onClick={onLogout}>Logout</Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome to the AI Healthcare System
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Advanced multimodal disease prediction using Vision Transformers, Clinical BERT, and Explainable AI
        </Typography>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Upload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  New Prediction
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload medical images (MRI/X-Ray), enter symptoms, and get AI-powered disease predictions with explainability.
                </Typography>
              </CardContent>
              <CardActions>
                <Button 
                  size="large" 
                  variant="contained" 
                  onClick={() => navigate('/prediction')}
                >
                  Start Analysis
                </Button>
              </CardActions>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Assessment sx={{ fontSize: 48, color: 'secondary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  View Reports
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Access previously generated clinical reports with AI explanations and retrieved medical references.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="large" variant="outlined" disabled>
                  Coming Soon
                </Button>
              </CardActions>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Description sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  System Status
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Backend: Connected<br/>
                  Models: Loaded<br/>
                  RAG: Ready
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
          <Typography variant="body2" align="center">
            <strong>⚠️ MEDICAL DISCLAIMER:</strong> This system is a research prototype for clinical decision support ONLY. 
            It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
            All predictions must be verified by licensed healthcare professionals.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}

export default Dashboard;