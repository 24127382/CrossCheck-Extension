// Background service worker

chrome.runtime.onInstalled.addListener(() => {
  console.log('CrossCheck extension installed');
  
  // Create context menu
  chrome.contextMenus.create({
    id: 'factcheck-context',
    title: 'Fact Check with CrossCheck',
    contexts: ['selection'],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'factcheck-context' && tab?.id) {
    console.log('[Background] Context menu clicked with text:', info.selectionText);
    
    // Open popup with the selected text
    const popupUrl = chrome.runtime.getURL('popup.html');
    
    // Store the selected text in storage for the popup to retrieve
    chrome.storage.local.set(
      { pendingFactCheck: info.selectionText },
      () => {
        // Open the popup
        chrome.action.openPopup();
      }
    );
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'factcheck') {
    handleFactCheck(request.data).then(sendResponse);
  }
  return true; // Will respond asynchronously
});

async function handleFactCheck(data: any) {
  try {
    const response = await fetch('http://localhost:8000/api/factcheck', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return await response.json();
  } catch (error) {
    console.error('Factcheck error:', error);
    return { error: 'Failed to fact-check' };
  }
}
