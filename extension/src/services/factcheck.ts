import { FactCheckRequest, FactCheckResponse } from '../types';
import { apiService } from './api';
import { storageService } from './storage';

export const factcheckService = {
  async checkClaim(text: string, context?: string): Promise<FactCheckResponse> {
    const request: FactCheckRequest = {
      text,
      context,
    };

    const result = await apiService.factCheck(request);
    await storageService.saveResult(text, result);
    return result;
  },
};
