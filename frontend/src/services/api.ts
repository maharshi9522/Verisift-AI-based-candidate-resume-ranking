const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(status: number, message: string, data?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export const TOKEN_STORAGE_KEY = 'verisift_auth_token';

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setStoredToken(token: string): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function removeStoredToken(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
}

export async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const token = getStoredToken();

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  if (token && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (response.status === 204) {
    return {} as T;
  }

  let responseData: any;
  try {
    responseData = await response.json();
  } catch {
    responseData = null;
  }

  if (!response.ok) {
    let message = 'An unexpected error occurred.';
    if (responseData && typeof responseData.detail === 'string') {
      message = responseData.detail;
    } else if (responseData && Array.isArray(responseData.detail)) {
      message = responseData.detail.map((err: any) => err.msg || err.message).join(', ');
    } else if (responseData && responseData.message) {
      message = responseData.message;
    }

    // Auto-clear invalid/expired token on 401
    if (response.status === 401 && endpoint !== '/api/auth/login') {
      removeStoredToken();
    }

    throw new ApiError(response.status, message, responseData);
  }

  return responseData as T;
}
