import { request, setStoredToken, removeStoredToken, getStoredToken } from './api';
import type { AuthResponse, LoginPayload, SignupPayload, LogoutResponse } from '../types/auth';
import type { User } from '../types/user';

export const authService = {
  async signup(payload: SignupPayload): Promise<AuthResponse> {
    const res = await request<AuthResponse>('/api/auth/signup', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (res.access_token) {
      setStoredToken(res.access_token);
    }
    return res;
  },

  async login(payload: LoginPayload): Promise<AuthResponse> {
    const res = await request<AuthResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (res.access_token) {
      setStoredToken(res.access_token);
    }
    return res;
  },

  async getCurrentUser(): Promise<User> {
    return request<User>('/api/auth/me', {
      method: 'GET',
    });
  },

  async logout(): Promise<LogoutResponse> {
    try {
      const res = await request<LogoutResponse>('/api/auth/logout', {
        method: 'POST',
      });
      return res;
    } finally {
      removeStoredToken();
    }
  },

  isAuthenticated(): boolean {
    return !!getStoredToken();
  },

  async checkHealth(): Promise<{ status: string; service: string }> {
    return request<{ status: string; service: string }>('/api/health');
  }
};
