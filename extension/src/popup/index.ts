// import { getSelectedText } from '../utils';
import { factcheckService } from '../services/factcheck';

const app = document.getElementById('app');

function displayDebugLog(message: string) {
  if (!app) return;
  const timestamp = new Date().toLocaleTimeString();
  const debugElement = app.querySelector('.debug-log');
  if (debugElement) {
    debugElement.innerHTML += `<div>[${timestamp}] ${message}</div>`;
  }
}

function displayLoading() {
  if (!app) return;
  app.innerHTML = `
    <div class="debug-log" style="font-size: 11px; color: #666; max-height: 100px; overflow-y: auto; margin-bottom: 8px; padding: 4px; border: 1px solid #ddd; font-family: monospace;"></div>
    <div class="loading">
      <div class="spinner"></div>
      Loading...
    </div>
  `;
  displayDebugLog('Loading started...');
}

function displayResult(result: any) {
  if (!app) return;
  app.innerHTML = `
    <div class="debug-log" style="font-size: 11px; color: #666; max-height: 80px; overflow-y: auto; margin-bottom: 8px; padding: 4px; border: 1px solid #ddd; font-family: monospace;"></div>
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
  displayDebugLog('Result displayed');
}

function displayError(message: string) {
  if (!app) return;
  app.innerHTML = `
    <div class="debug-log" style="font-size: 11px; color: #cc0000; max-height: 100px; overflow-y: auto; margin-bottom: 8px; padding: 4px; border: 1px solid #ddd; font-family: monospace;"></div>
    <div class="error-message">${message}</div>
  `;
}

async function initializePopup() {
  try {
    console.log('[Popup] ===== POPUP INIT START =====');
    
    let selectedText: string | null = null;
    
    // First, check if text came from context menu
    console.log('[Popup] Step 1: Checking for context menu text...');
    const stored = await chrome.storage.local.get('pendingFactCheck');
    if (stored.pendingFactCheck) {
      console.log('[Popup] Step 1b: ✓ Found text from context menu!');
      selectedText = stored.pendingFactCheck;
      // Clear the stored text
      await chrome.storage.local.remove('pendingFactCheck');
      console.log('[Popup] Step 1c: Cleared storage');
    } else {
      // No context menu text, try getting selected text from content script
      console.log('[Popup] Step 2: No context menu text, getting from content script...');
      
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) {
        throw new Error('No active tab found');
      }
      console.log('[Popup] Step 2b: Active tab found:', tab.url);
      
      // Check if tab URL is restricted (chrome://, extension://, etc.)
      if (!tab.url || tab.url.startsWith('chrome://') || tab.url.startsWith('edge://') || 
          tab.url.startsWith('moz-extension://') || tab.url.startsWith('chrome-extension://')) {
        throw new Error('RESTRICTED_CONTEXT');
      }
      
      console.log('[Popup] Step 3: Sending message to tab...');
      
      // Create a promise with timeout - send to the specific tab's content script
      const messagePromise = chrome.tabs.sendMessage(tab.id, { action: 'getSelectedText' });
      console.log('[Popup] Step 4: Message sent, waiting for response...');
      
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Message timeout - content script may not be loaded')), 5000)
      );
      
      const response = await Promise.race([messagePromise, timeoutPromise]);
      console.log('[Popup] Step 5: Response received:', response);
      
      selectedText = response?.text;
      console.log('[Popup] Step 6: Selected text extracted:', selectedText);
    }

    if (!selectedText?.trim()) {
      console.log('[Popup] Step 7: No text selected');
      if (app) {
        app.innerHTML = '<div class="debug-log" style="font-size: 11px; color: #666; padding: 4px; border: 1px solid #ddd; font-family: monospace;"><div>[✓] Content script loaded</div><div>[✗] No text selected</div></div><p>Please select text and try again, or use right-click menu</p>';
      }
      return;
    }

    console.log('[Popup] Step 8: Starting display loading...');
    displayLoading();
    displayDebugLog('✓ Ready to fact-check');
    displayDebugLog('Text: ' + selectedText.substring(0, 50) + '...');

    try {
      console.log('[Popup] Step 9: Sending to backend...');
      displayDebugLog('Sending to backend...');
      const result = await factcheckService.checkClaim(selectedText);
      console.log('[Popup] Step 10: Backend response received:', result);
      displayDebugLog('Result received from backend');
      displayResult(result);
    } catch (error: any) {
      console.error('[Popup] Step 10-ERROR: Backend error:', error);
      displayDebugLog('✗ Backend error: ' + error.message);
      displayError('Backend unavailable');
    }
  } catch (error: any) {
    console.error('[Popup] INIT ERROR:', error);
    console.log('[Popup] Error details:', {
      message: error.message,
      stack: error.stack,
      isTimeout: error.message.includes('timeout')
    });
    
    if (app) {
      let errorMsg = '[✗] Error: ' + error.message;
      
      if (error.message === 'RESTRICTED_CONTEXT') {
        errorMsg = '[✗] Cannot access this page type. Please use the extension on regular websites.';
      } else if (error.message.includes('Could not establish connection') || error.message.includes('Receiving end does not exist')) {
        errorMsg = '[✗] Content script blocked (likely due to site security policy). Try on a news site or regular webpage instead.';
      } else if (error.message.includes('timeout')) {
        errorMsg = '[✗] Content script not loaded. Make sure you\'re on a regular webpage (not ChatGPT, Gmail, etc.).';
      }
      
      app.innerHTML = '<div class="debug-log" style="font-size: 11px; color: #cc0000; padding: 4px; border: 1px solid #ddd; font-family: monospace;"><div>' + errorMsg + '</div><div style="font-size: 10px; margin-top: 4px; color: #999;">Try: BBC.com, Wikipedia, or any news site</div></div><p style="margin-top: 8px; font-size: 13px;">Select text on a regular webpage or use right-click menu</p>';
    }
  }
}

initializePopup();
