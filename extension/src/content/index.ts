// Content script - runs in webpage context

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getSelectedText') {
    const selectedText = window.getSelection()?.toString();
    sendResponse({ text: selectedText });
  } else if (request.action === 'getPageContext') {
    const context = {
      title: document.title,
      url: window.location.href,
      text: document.body.innerText.substring(0, 5000),
    };
    sendResponse(context);
  }
});

// Highlight fact-check results
export function highlightResult(element: HTMLElement, verdict: string) {
  const color = verdict === 'REFUTED' ? '#ffebee' : '#e8f5e9';
  element.style.backgroundColor = color;
  element.style.borderLeft = `4px solid ${verdict === 'REFUTED' ? '#c62828' : '#2e7d32'}`;
  element.style.padding = '4px 8px';
}
