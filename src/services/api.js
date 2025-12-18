import axios from 'axios';

// Point this to your FastAPI backend
const API_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost2:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const endpoints = {
  history: (id) => `/sales/history/${id}`,
  forecast: (id) => `/sales/forecast/${id}`,
  live: (id) => `/sales/forecast/live/${id}`,
  metrics: '/metrics/model',
};