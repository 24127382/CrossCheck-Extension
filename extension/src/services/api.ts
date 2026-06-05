import axios from 'axios';
import { FactCheckRequest, FactCheckResponse, ExplainResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiService = {
  async factCheck(request: FactCheckRequest): Promise<FactCheckResponse> {
    try {
      const response = await axios.post<FactCheckResponse>(
        `${API_BASE_URL}/api/factcheck`,
        request,
        { timeout: 30000 }
      );
      return response.data;
    } catch (error: any) {
      if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
        throw new Error('Backend unavailable');
      }
      throw error;
    }
  },

  async explain(claim: string): Promise<ExplainResponse> {
    try {
      const response = await axios.post<ExplainResponse>(
        `${API_BASE_URL}/api/explain`,
        { claim },
        { timeout: 30000 }
      );
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 400) {
        throw new Error('No cached result found. Please run fact-check first.');
      }
      if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
        throw new Error('Backend unavailable');
      }
      throw error;
    }
  },

  async health(): Promise<{ status: string }> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/health`, {
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      throw new Error('Backend unavailable');
    }
  },
};


