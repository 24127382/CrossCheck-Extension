// Content script - runs in webpage context

console.log('[Content Script] Content script loaded and ready');

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[Content Script] Message received:', request, 'from sender:', sender);
  
  try {
    if (request.action === 'getSelectedText') {
      console.log('[Content Script] Getting selected text...');
      const selectedText = window.getSelection()?.toString();
      console.log('[Content Script] Selected text result:', selectedText);
      console.log('[Content Script] Sending response back...');
      sendResponse({ text: selectedText || '' });
      console.log('[Content Script] Response sent');
    } else if (request.action === 'getPageContext') {
      console.log('[Content Script] Getting page context...');
      const context = {
        title: document.title,
        url: window.location.href,
        text: document.body.innerText.substring(0, 5000),
      };
      sendResponse(context);
    } else {
      console.log('[Content Script] Unknown action:', request.action);
      sendResponse({ error: 'Unknown action' });
    }
  } catch (error: any) {
    console.error('[Content Script] Error handling message:', error);
    sendResponse({ error: error.message });
  }
  
  // Return true to indicate we'll send response asynchronously
  return true;
});

// Highlight fact-check results
export function highlightResult(element: HTMLElement, verdict: string) {
  const color = verdict === 'REFUTED' ? '#ffebee' : '#e8f5e9';
  element.style.backgroundColor = color;
  element.style.borderLeft = `4px solid ${verdict === 'REFUTED' ? '#c62828' : '#2e7d32'}`;
  element.style.padding = '4px 8px';
}
