import type { User } from './user';

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface SignupPayload {
  full_name: string;
  email: string;
  password: string;
  confirm_password?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface LogoutResponse {
  message: string;
  success: boolean;
}
