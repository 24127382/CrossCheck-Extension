export function getSelectedText(): string | null {
  return window.getSelection()?.toString() || null;
}

export function truncateText(text: string, maxLength: number = 100): string {
  return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

export function formatConfidence(confidence: number): string {
  return (confidence * 100).toFixed(1) + '%';
}
