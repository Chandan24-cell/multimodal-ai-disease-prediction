import axios from 'axios';

// Render serves this build and FastAPI from the same origin in production.
// Keep a direct backend URL for the local Vite development server.
const API_BASE_URL = import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000/api' : '/api');

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add JWT token to requests automatically
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth Service
export const authService = {
  login: async (username, password) => {
    try {
      const response = await axios.post(
        `${API_BASE_URL}/auth/login`,
        `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }
      );

      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        return response.data;
      }
      throw new Error('No token received');
    } catch (error) {
      console.error('Login error details:', error.response?.data || error.message);
      throw error;
    }
  },

  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },
};

// Other services...
export const predictionService = {
  predict: async (imageFile, symptomText, patientData) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('symptom_text', symptomText);
    formData.append('patient_data_json', JSON.stringify(patientData));

    const response = await api.post('/predict/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },
};

export const reportService = {
  generateReport: async (patientData, predictions) => {
    const response = await api.post('/reports/generate', {
      patient_data: patientData,
      predictions: predictions,
    });
    return response.data;
  },

  getReport: async (reportId) => {
    const response = await api.get(`/reports/${reportId}`);
    return response.data;
  },
};

export const patientService = {
  createPatient: async (patientData) => {
    const response = await api.post('/patients/', patientData);
    return response.data;
  },

  listPatients: async () => {
    const response = await api.get('/patients/');
    return response.data;
  },
};

export default api;
