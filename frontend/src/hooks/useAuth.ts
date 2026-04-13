/**
 * Custom hook for authentication state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import authClient from '../lib/authClient';

export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  role: 'OPERATOR' | 'ANALYST' | 'ADMIN';
  is_active: number;
  created_at: string;
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string, fullName: string, inviteToken: string) => Promise<void>;
  logout: () => Promise<void>;
  fetchCurrentUser: () => Promise<void>;
  refreshTokens: () => Promise<void>;
}

const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isLoading: false,
      isAuthenticated: false,

      login: async (username: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await authClient.post('/api/auth/login', {
            username,
            password,
          });

          const { access_token, refresh_token, user } = response.data;
          localStorage.setItem('access_token', access_token);
          localStorage.setItem('refresh_token', refresh_token);

          set({
            user,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw error.response?.data?.detail || 'Login failed';
        }
      },

      register: async (username: string, email: string, password: string, fullName: string, inviteToken: string) => {
        set({ isLoading: true });
        try {
          const response = await authClient.post('/api/auth/register', {
            username,
            email,
            password,
            full_name: fullName,
            invite_token: inviteToken,
          });

          const { access_token, refresh_token, user } = response.data;
          localStorage.setItem('access_token', access_token);
          localStorage.setItem('refresh_token', refresh_token);

          set({
            user,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw error.response?.data?.detail || 'Registration failed';
        }
      },

      logout: async () => {
        try {
          const refreshToken = localStorage.getItem('refresh_token');
          if (refreshToken) {
            await authClient.post('/api/auth/logout', {
              refresh_token: refreshToken,
            });
          }
        } finally {
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          set({
            user: null,
            isAuthenticated: false,
          });
        }
      },

      fetchCurrentUser: async () => {
        set({ isLoading: true });
        try {
          const response = await authClient.get('/api/auth/me');
          set({
            user: response.data,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch {
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      },

      refreshTokens: async () => {
        try {
          const refreshToken = localStorage.getItem('refresh_token');
          if (!refreshToken) throw new Error('No refresh token');

          const response = await authClient.post('/api/auth/refresh', {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token: newRefreshToken } = response.data;
          localStorage.setItem('access_token', access_token);
          localStorage.setItem('refresh_token', newRefreshToken);
        } catch (error) {
          get().logout();
          throw error;
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

export const useAuth = () => useAuthStore();
