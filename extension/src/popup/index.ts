// import { getSelectedText } from '../utils';
import { factcheckService } from '../services/factcheck';

const app = document.getElementById('app');

async function initializePopup() {
  console.log('Popup initialized');
  // const selectedText = await chrome.tabs.executeScript({
  //   code: `(${getSelectedText.toString()})()`,
  // });
  const selectedText = "The Earth is flat";
  if (selectedText) {
    const text = selectedText;
    console.log('Checking claim:', text);

    try {
      console.log('Calling factcheckService...');
      const result = await factcheckService.checkClaim(text);
      console.log('Result received:', result);
      displayResult(result);
    } catch (error: any) {
      console.error('Error:', error);
      displayError(`Failed to fact-check claim: ${error.message}`);
    }
  } else {
    if (app) app.innerHTML = '<p>Please select text to fact-check</p>';
  }
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
  if (app) app.innerHTML = `<p style="color: #d32f2f;">${message}</p>`;
}

initializePopup();
