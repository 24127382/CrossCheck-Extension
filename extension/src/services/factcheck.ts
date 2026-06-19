import { FactCheckRequest, FactCheckResponse, ExplainResponse } from '../types';
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

  async explainClaim(claim: string): Promise<ExplainResponse> {
    return await apiService.explain(claim);
  },

  async checkWithLLM(text: string): Promise<FactCheckResponse> {
    const result = await apiService.llmCheck(text);
    await storageService.saveResult(text, result);  // Lưu cache
    return result;
}
};
