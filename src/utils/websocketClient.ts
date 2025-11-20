/**
 * WebSocket client for chat completions
 * This replaces the HTTP streaming endpoint with a WebSocket connection
 */

// Get the server base URL from environment or use default
const SERVER_BASE_URL = process.env.NEXT_PUBLIC_SERVER_BASE_URL || 'http://localhost:8001';

// Convert HTTP URL to WebSocket URL
const getWebSocketUrl = () => {  
  const baseUrl = SERVER_BASE_URL;
  // Replace http:// with ws:// or https:// with wss://
  const wsBaseUrl = baseUrl.replace(/^http/, 'ws');
  return `${wsBaseUrl}/ws/chat`;
};

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatCompletionRequest {
  repo_url: string;
  messages: ChatMessage[];
  filePath?: string;
  token?: string;
  type?: string;
  provider?: string;
  model?: string;
  custom_model?: string;
  language?: string;
  excluded_dirs?: string;
  excluded_files?: string;
  included_dirs?: string;
  included_files?: string;
  deep_research?: boolean;
  max_iterations?: number;
}

/**
 * Creates a WebSocket connection for chat completions
 * @param request The chat completion request
 * @param onMessage Callback for received messages
 * @param onError Callback for errors
 * @param onClose Callback for when the connection closes
 * @returns The WebSocket connection
 */
export const createChatWebSocket = (
  request: ChatCompletionRequest,
  onMessage: (message: string) => void,
  onError: (error: Event) => void,
  onClose: () => void
): Promise<WebSocket> => {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(getWebSocketUrl());

    ws.onopen = () => {
      console.log("WebSocket connection established");
      ws.send(JSON.stringify(request));
      resolve(ws);
    };

    ws.onmessage = (event) => onMessage(event.data);

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      onError(error);
      reject(error);
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed");
      onClose();
    };
  });
};

/**
 * Closes a WebSocket connection
 * @param ws The WebSocket connection to close
 */
export const closeWebSocket = (ws: WebSocket | null): void => {
  if (!ws) return;
  if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
    ws.close();
  }
};
