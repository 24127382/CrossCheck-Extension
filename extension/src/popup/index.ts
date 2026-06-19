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
    <div class="debug-log" style="font-size: 11px; color: #666; max-height: 100px; overflow-y: auto; margin-bottom: 8px; padding: 4px; border: 1px solid #ddd; font-family: monospace;"></div>
    <div class="result-card">
      <div class="verdict ${result.verdict.toLowerCase()}">
        ${result.verdict}
      </div>
      <div class="confidence">
        Confidence: ${(result.confidence * 100).toFixed(1)}%
      </div>
      <p style="margin-top: 8px; font-size: 14px;">${result.summary}</p>
      <button id="explainBtn" style="margin-top: 12px; padding: 8px 12px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer;">
        Get LLM Explanation
      </button>
      <div id="explanationContainer" style="margin-top: 12px; font-size: 13px; color: #333; line-height: 1.4; display: none; padding: 8px; background: #f9f9f9; border-radius: 4px;"></div>
    </div>
  `;
  displayDebugLog('Result displayed');
  
  // Add click handler
  document.getElementById('explainBtn')?.addEventListener('click', async () => {
    try {
      displayDebugLog('Calling LLM...');
      const explanation = await factcheckService.explainClaim(result.claim);
      
      // SỬA LỖI 1: Sử dụng biến explanation để hiển thị lên UI
      const container = document.getElementById('explanationContainer');
      if (container) {
        container.innerHTML = `<strong>Explanation:</strong><br/>${explanation}`;
        container.style.display = 'block';
      }
      displayDebugLog('Explanation displayed');
    } catch (error: any) {
      displayDebugLog('Error: ' + error.message);
    }
  });
}

function displayError(message: string) {
  if (!app) return;
  app.innerHTML = `
    <div class="debug-log" style="font-size: 11px; color: #cc0000; max-height: 100px; overflow-y: auto; margin-bottom: 8px; padding: 4px; border: 1px solid #ddd; font-family: monospace;"></div>
    <div class="error-message">${message}</div>
  `;
}

function displayInputForm() {
  if (!app) return;
  app.innerHTML = `
    <div class="input-form">
      <textarea id="claimInput" placeholder="Enter claim to fact-check..." rows="4"></textarea>
      <div class="button-group">
        <button id="submitBtn">Fact Check</button>
        <button id="llmCheckBtn">Check with LLM</button>
      </div>
    </div>
  `;
  
  const handleCheck = async (isLLMOnly: boolean) => {
    const text = (document.getElementById('claimInput') as HTMLTextAreaElement).value;
    if (text.trim()) {
      displayLoading();
      try {
        const result = isLLMOnly 
          ? await factcheckService.checkWithLLM(text)
          : await factcheckService.checkClaim(text);
        displayResult(result);  // Cùng function displayResult cho cả hai
      } catch (error: any) {
        const errorMsg = isLLMOnly ? 'LLM check failed' : 'Backend unavailable';
        displayError(`${errorMsg}: ${error.message}`);
      }
    }
  };
  
  document.getElementById('submitBtn')?.addEventListener('click', () => handleCheck(false));
  document.getElementById('llmCheckBtn')?.addEventListener('click', () => handleCheck(true));
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
      console.log('[Popup] Step 7: No text selected. Showing input form...');
      // SỬA LỖI 2: Gọi hàm displayInputForm() tại đây thay vì hiện thông báo lỗi rườm rà
      displayInputForm();
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
        errorMsg = '[✗] Cannot access this page type. Showing input form...';
        // Nếu trang bị cấm (như chrome://), cho người dùng nhập tay luôn
        displayInputForm();
        return;
      } else if (error.message.includes('Could not establish connection') || error.message.includes('Receiving end does not exist')) {
        errorMsg = '[✗] Content script blocked. Showing input form...';
        displayInputForm();
        return;
      } else if (error.message.includes('timeout')) {
        errorMsg = '[✗] Content script not loaded. Showing input form...';
        displayInputForm();
        return;
      }
      
      app.innerHTML = '<div class="debug-log" style="font-size: 11px; color: #cc0000; padding: 4px; border: 1px solid #ddd; font-family: monospace;"><div>' + errorMsg + '</div></div>';
    }
  }
}

initializePopup();