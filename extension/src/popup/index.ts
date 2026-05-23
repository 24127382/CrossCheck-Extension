// import { getSelectedText } from '../utils';
import { factcheckService } from '../services/factcheck';

const app = document.getElementById('app');

function displayLoading() {
  if (!app) return;
  app.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      Loading...
    </div>
  `;
}

function displayResult(result: any) {
  if (!app) return;
  app.innerHTML = `
    <div class="result-card">
      <div class="verdict ${result.verdict.toLowerCase()}">
        ${result.verdict}
      </div>
      <div class="confidence">
        Confidence: ${(result.confidence * 100).toFixed(1)}%
      </div>
      <p style="margin-top: 8px; font-size: 14px;">${result.summary}</p>
    </div>
  `;
}

function displayError(message: string) {
  if (!app) return;
  app.innerHTML = `<div class="error-message">${message}</div>`;
}

async function initializePopup() {
  // const selectedText = await chrome.tabs.executeScript({
  //   code: `(${getSelectedText.toString()})()`,
  // });
  const selectedText = "The Earth is flat";
  if (selectedText) {
    const text = selectedText;
    displayLoading();

    try {
      const result = await factcheckService.checkClaim(text);
      displayResult(result);
    } catch (error: any) {
      // Handle network/backend errors gracefully
      if (error.code === 'ECONNREFUSED' || error.message.includes('Network') || !navigator.onLine) {
        displayError('Backend unavailable');
      } else {
        displayError('Backend unavailable');
      }
    }
  } else {
    if (app) app.innerHTML = '<p>Please select text to fact-check</p>';
  }
}

initializePopup();
