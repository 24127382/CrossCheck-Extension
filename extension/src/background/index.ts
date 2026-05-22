// Background service worker

chrome.runtime.onInstalled.addListener(() => {
  console.log('CrossCheck extension installed');
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
