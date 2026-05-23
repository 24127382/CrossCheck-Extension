// Global declarations
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      REACT_APP_API_URL?: string;
    }
  }
}

// API Response Types
export interface FactCheckResponse {
  claim: string;
  verdict: 'REFUTED' | 'SUPPORTED' | 'NOT_ENOUGH_INFO' | 'DISPUTED';
  confidence: number;
  summary: string;
  evidences: Evidence[];
}

export interface Evidence {
  source: string;
  stance: 'supports' | 'contradicts' | 'neutral';
  score: number;
  text?: string;
}

// Message Types
export interface FactCheckRequest {
  text: string;
  context?: string;
}

export interface StorageData {
  history: FactCheckHistory[];
  apiKey?: string;
}

export interface FactCheckHistory {
  claim: string;
  result: FactCheckResponse;
  timestamp: number;
}
