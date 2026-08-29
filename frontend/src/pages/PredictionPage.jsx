// frontend/src/pages/PredictionPage.jsx
import React, { useState } from 'react';
import {
  Container, Grid, Paper, Typography, TextField, Button, Box,
  Card, CardContent, Alert, Stepper, Step, StepLabel, LinearProgress
} from '@mui/material';
import { CloudUpload, Psychology, Analytics } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { predictionService, reportService } from '../services/api';
import ConfidenceChart from '../charts/ConfidenceChart';
import HeatmapViewer from '../charts/HeatmapViewer';
import TextAttentionViewer from '../charts/TextAttentionViewer';

function PredictionPage() {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [symptomText, setSymptomText] = useState('');
  const [patientData, setPatientData] = useState({
    age: '',
    gender: 'male',
    vitals: { heart_rate: '', systolic_bp: '', diastolic_bp: '', temperature: '', spo2: '' },
    prior_conditions: []
  });
  const [predictions, setPredictions] = useState(null);
  const [reportId, setReportId] = useState(null);
  const [error, setError] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handlePatientDataChange = (field, value) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      setPatientData(prev => ({
        ...prev,
        [parent]: { ...prev[parent], [child]: value }
      }));
    } else {
      setPatientData(prev => ({ ...prev, [field]: value }));
    }
  };

  const handleSubmit = async () => {
    if (!imageFile || !symptomText || !patientData.age) {
      setError('Please fill in all required fields and upload an image.');
      return;
    }

    setLoading(true);
    setError('');
    setActiveStep(1);

    try {
      const result = await predictionService.predict(imageFile, symptomText, patientData);
      setPredictions(result);
      setActiveStep(2);
    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    setLoading(true);
    try {
      const report = await reportService.generateReport(patientData, predictions);
      setReportId(report.report_id);
      navigate(`/report/${report.report_id}`);
    } catch (err) {
      setError('Report generation failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const steps = ['Input Data', 'Processing', 'Results'];

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Multimodal Disease Prediction
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}><StepLabel>{label}</StepLabel></Step>
        ))}
      </Stepper>

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      {loading && <LinearProgress sx={{ mb: 3 }} />}

      <Grid container spacing={3}>
        {/* Input Section */}
        {activeStep === 0 && (
          <>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <CloudUpload sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Upload Medical Image
                </Typography>
                <input
                  accept="image/*,.dcm,.dicom"
                  style={{ display: 'none' }}
                  id="image-upload"
                  type="file"
                  onChange={handleImageChange}
                />
                <label htmlFor="image-upload">
                  <Button variant="contained" component="span" fullWidth sx={{ mt: 2 }}>
                    Choose Medical Image (DICOM, PNG, JPG)
                  </Button>
                </label>
                {imagePreview && (
                  <Box sx={{ mt: 2, textAlign: 'center' }}>
                    <img src={imagePreview} alt="Preview" style={{ maxWidth: '100%', maxHeight: 300, borderRadius: 8 }} />
                  </Box>
                )}
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <Psychology sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Patient Information
                </Typography>
                <TextField fullWidth label="Age" type="number" margin="normal" 
                  value={patientData.age} onChange={(e) => handlePatientDataChange('age', e.target.value)} />
                <TextField fullWidth label="Gender" select SelectProps={{ native: true }} margin="normal"
                  value={patientData.gender} onChange={(e) => handlePatientDataChange('gender', e.target.value)}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </TextField>
                <TextField fullWidth label="Symptoms" multiline rows={4} margin="normal"
                  value={symptomText} onChange={(e) => setSymptomText(e.target.value)}
                  placeholder="Describe patient symptoms..." />
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Vitals</Typography>
                <Grid container spacing={2}>
                  {['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2'].map((vital) => (
                    <Grid item xs={12} sm={4} key={vital}>
                      <TextField fullWidth label={vital.replace('_', ' ').toUpperCase()} type="number"
                        value={patientData.vitals[vital]}
                        onChange={(e) => handlePatientDataChange(`vitals.${vital}`, e.target.value)} />
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>

            <Grid item xs={12} sx={{ textAlign: 'center' }}>
              <Button variant="contained" size="large" onClick={handleSubmit} disabled={loading}>
                Run AI Prediction
              </Button>
            </Grid>
          </>
        )}

        {/* Results Section */}
        {activeStep === 2 && predictions && (
          <>
            <Grid item xs={12}>
              <Alert severity="info" sx={{ mb: 2 }}>
                <strong>Predicted Diseases:</strong> {predictions.final_diseases.join(', ')} 
                (Confidence: {(predictions.confidence_score * 100).toFixed(1)}%)
              </Alert>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Prediction Confidence
                </Typography>
                <ConfidenceChart predictions={predictions.fused_prediction} />
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Explainability</Typography>
                {predictions.explainability.image_heatmap && (
                  <HeatmapViewer heatmapBase64={predictions.explainability.image_heatmap} />
                )}
                {predictions.explainability.text_attention && (
                  <TextAttentionViewer tokens={predictions.explainability.text_attention} />
                )}
              </Paper>
            </Grid>

            <Grid item xs={12} sx={{ textAlign: 'center' }}>
              <Button variant="contained" size="large" onClick={handleGenerateReport} disabled={loading} sx={{ mr: 2 }}>
                Generate Clinical Report
              </Button>
              <Button variant="outlined" size="large" onClick={() => { setActiveStep(0); setPredictions(null); }}>
                New Prediction
              </Button>
            </Grid>
          </>
        )}
      </Grid>
    </Container>
  );
}

export default PredictionPage;