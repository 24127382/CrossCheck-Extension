import { StorageData, FactCheckHistory, FactCheckResponse } from '../types';

export const storageService = {
  async getHistory(): Promise<FactCheckHistory[]> {
    const data = await chrome.storage.local.get('history');
    return data.history || [];
  },

  async saveResult(claim: string, result: FactCheckResponse): Promise<void> {
    const history = await this.getHistory();
    history.push({
      claim,
      result,
      timestamp: Date.now(),
    });
    await chrome.storage.local.set({ history });
  },

  async clearHistory(): Promise<void> {
    await chrome.storage.local.remove('history');
  },

  async getApiKey(): Promise<string | undefined> {
    const data = await chrome.storage.sync.get('apiKey');
    return data.apiKey;
  },

  async setApiKey(key: string): Promise<void> {
    await chrome.storage.sync.set({ apiKey: key });
  },
};
