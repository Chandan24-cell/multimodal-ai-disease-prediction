import React, { useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Container,
  Paper,
  TextField,
  Typography,
} from '@mui/material';
import { authService } from '../services/api';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setLoading(true);

    try {
      await authService.login(username, password);
      onLogin();
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Healthcare AI Login
        </Typography>
        <Typography color="text.secondary" sx={{ mb: 3 }}>
          Sign in to access the clinical decision-support tools.
        </Typography>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <Box component="form" onSubmit={handleSubmit}>
          <TextField
            fullWidth
            required
            label="Username"
            margin="normal"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
          />
          <TextField
            fullWidth
            required
            label="Password"
            type="password"
            margin="normal"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          <Button
            fullWidth
            type="submit"
            variant="contained"
            disabled={loading}
            sx={{ mt: 3 }}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default Login;
