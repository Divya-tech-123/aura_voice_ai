/**
 * AURA Frontend API Service
 *
 * Encapsulates HTTP communication with the FastAPI backend.
 * Keeps API requests, error handling, and URL configuration decoupled from UI components.
 */

import { ActivityItem } from '../components/AgentActivity';

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  response: string;
  activities?: ActivityItem[];
}

export interface VoiceResponse {
  user_text: string;
  response: string;
  audio: string | null;
  audio_format?: string;
  activities?: ActivityItem[];
}

export interface HealthResponse {
  status: string;
  app: string;
  version?: string;
}

// Read backend URL from Vite environment variables (fallback to localhost:8000 for local dev)
const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

/**
 * Send a user message to the AURA backend agent.
 *
 * @param message The user query or instruction string
 * @returns The agent's structured response including text and activities
 * @throws Error with a user-friendly message on failure
 */
export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const cleanMessage = message.trim();
  if (!cleanMessage) {
    throw new Error('Please enter a message before sending.');
  }

  const endpoint = `${API_BASE_URL}/api/chat`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ message: cleanMessage }),
    });

    if (!response.ok) {
      let errorMessage = `Server responded with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((item: any) => item.msg || item).join('; ');
        } else if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch {
        // Response body was not JSON
      }
      throw new Error(errorMessage);
    }

    const data: ChatResponse = await response.json();
    if (!data || typeof data.response !== 'string') {
      throw new Error('Received unexpected or malformed response format from AURA backend.');
    }

    return data;
  } catch (error: any) {
    // Gracefully identify connection/network errors
    if (
      error.name === 'TypeError' ||
      (error.message && (error.message.includes('fetch') || error.message.includes('NetworkError')))
    ) {
      throw new Error(
        `Unable to connect to AURA backend at ${API_BASE_URL}. Please ensure the backend server is running.`
      );
    }

    // Re-throw formatted message
    throw error;
  }
}

/**
 * Check backend service health and readiness.
 *
 * @returns true if backend is reachable and healthy, false otherwise
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) return false;
    const data: HealthResponse = await response.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

/**
 * Send recorded audio blob to the AURA backend voice pipeline.
 *
 * @param audioBlob The captured audio Blob (typically audio/wav)
 * @returns VoiceResponse containing recognized user text, assistant reply, and audio URI
 * @throws Error with a user-friendly message on failure
 */
export async function sendVoiceAudio(audioBlob: Blob): Promise<VoiceResponse> {
  if (!audioBlob || audioBlob.size === 0) {
    throw new Error('Recorded audio is empty. Please speak into the microphone and try again.');
  }

  const resolvedContentType = audioBlob.type || 'audio/wav';
  console.debug('[Voice API] Sending recorded audio:', {
    type: resolvedContentType,
    size: `${audioBlob.size} bytes`,
  });

  const endpoint = `${API_BASE_URL}/api/voice`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': resolvedContentType,
        'Accept': 'application/json',
      },
      body: audioBlob,
    });

    if (!response.ok) {
      let errorMessage = `Voice server responded with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((item: any) => item.msg || item).join('; ');
        } else if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch {
        // Non-JSON response
      }
      console.warn('[Voice API] Voice request failed:', response.status, errorMessage);
      throw new Error(errorMessage);
    }

    const data: VoiceResponse = await response.json();
    if (!data || typeof data.user_text !== 'string' || typeof data.response !== 'string') {
      throw new Error('Received unexpected or malformed response format from AURA voice backend.');
    }

    console.debug('[Voice API] Voice response received:', {
      user_text: data.user_text,
      response_snippet: data.response.substring(0, 60) + '...',
      has_audio: !!data.audio,
      audio_format: data.audio_format,
    });

    return data;
  } catch (error: any) {
    if (
      error.name === 'TypeError' ||
      (error.message && (error.message.includes('fetch') || error.message.includes('NetworkError')))
    ) {
      throw new Error(
        `Unable to connect to AURA voice backend at ${API_BASE_URL}. Please ensure the backend server is running.`
      );
    }
    throw error;
  }
}

export interface DocumentItem {
  filename: string;
  status: string;
  size?: number;
}

export interface DocumentUploadResponse {
  success: boolean;
  filename: string;
  message: string;
  chunks_indexed?: number;
}

export interface DocumentListResponse {
  documents: DocumentItem[];
}

/**
 * Upload a document (PDF, DOCX, TXT) to the AURA knowledge base for RAG.
 *
 * @param file The document File to upload
 * @returns Structured result indicating indexing success
 * @throws Error on validation or indexing failure
 */
export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  if (!file) {
    throw new Error('Please select a document file to upload.');
  }

  // Basic client-side validation
  const lowerName = file.name.toLowerCase();
  const validExts = ['.pdf', '.docx', '.txt'];
  if (!validExts.some((ext) => lowerName.endsWith(ext))) {
    throw new Error('Unsupported file format. Please upload a PDF, DOCX, or TXT file.');
  }

  const MAX_SIZE = 10 * 1024 * 1024; // 10 MB
  if (file.size > MAX_SIZE) {
    throw new Error('File size exceeds the 10 MB limit.');
  }

  const endpoint = `${API_BASE_URL}/api/documents/upload`;
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `Server responded with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((item: any) => item.msg || item).join('; ');
        } else if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch {
        // Non-JSON response
      }
      throw new Error(errorMessage);
    }

    const data: DocumentUploadResponse = await response.json();
    return data;
  } catch (error: any) {
    if (
      error.name === 'TypeError' ||
      (error.message && (error.message.includes('fetch') || error.message.includes('NetworkError')))
    ) {
      throw new Error(
        `Unable to connect to AURA backend at ${API_BASE_URL}. Please ensure the server is running.`
      );
    }
    throw error;
  }
}

/**
 * Fetch list of indexed documents in AURA's knowledge base.
 *
 * @returns Array of DocumentItem objects
 */
export async function fetchDocuments(): Promise<DocumentItem[]> {
  const endpoint = `${API_BASE_URL}/api/documents`;

  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      return [];
    }

    const data: DocumentListResponse = await response.json();
    return data.documents || [];
  } catch (error) {
    console.warn('Could not fetch documents:', error);
    return [];
  }
}


