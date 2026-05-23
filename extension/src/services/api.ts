import axios from 'axios';
import { FactCheckRequest, FactCheckResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiService = {
  async factCheck(request: FactCheckRequest): Promise<FactCheckResponse> {
    const response = await axios.post<FactCheckResponse>(
      `${API_BASE_URL}/api/factcheck`,
      request
    );
    return response.data;
  },

  async health(): Promise<{ status: string }> {
    const response = await axios.get(`${API_BASE_URL}/api/health`);
    return response.data;
  },
};


