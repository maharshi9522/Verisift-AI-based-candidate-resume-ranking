import { request, getStoredToken } from './api';
import type {
  Candidate,
  CandidateListResponse,
  BatchUploadResponse,
  DeleteCandidateResponse,
  RankJobResponse,
} from '../types/candidate';

export const candidateService = {
  async exportCandidateShortlist(jobId: number, limit: number, format: 'xlsx' | 'pdf'): Promise<void> {
    const token = getStoredToken();
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    const url = `${API_BASE_URL}/api/jobs/${jobId}/candidates/export?limit=${limit}&format=${format}`;

    const headers: Record<string, string> = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, { method: 'GET', headers });

    if (!response.ok) {
      let errorMsg = `Export failed with status ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) errorMsg = errJson.detail;
      } catch {
        // ignore
      }
      throw new Error(errorMsg);
    }

    const blob = await response.blob();
    const disposition = response.headers.get('Content-Disposition');
    let filename = `Verisift_Shortlist_Job${jobId}_Top${limit}.${format}`;
    if (disposition && disposition.includes('filename=')) {
      const match = disposition.match(/filename="?([^"]+)"?/);
      if (match && match[1]) {
        filename = match[1];
      }
    }

    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  },
  async rankJobCandidates(jobId: number): Promise<RankJobResponse> {
    return request<RankJobResponse>(`/api/jobs/${jobId}/rank`, { method: 'POST' });
  },
  async uploadJobCandidates(jobId: number, files: File[]): Promise<BatchUploadResponse> {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    return request<BatchUploadResponse>(`/api/jobs/${jobId}/candidates/upload`, {
      method: 'POST',
      body: formData,
    });
  },

  async getJobCandidates(
    jobId: number,
    params?: { search?: string; status?: string; page?: number; pageSize?: number }
  ): Promise<CandidateListResponse> {
    const query = new URLSearchParams();
    if (params?.search && params.search.trim()) {
      query.append('search', params.search.trim());
    }
    if (params?.status && params.status !== 'all') {
      query.append('status', params.status);
    }
    if (params?.page) {
      query.append('page', String(params.page));
    }
    if (params?.pageSize) {
      query.append('page_size', String(params.pageSize));
    }

    const qs = query.toString();
    const endpoint = qs ? `/api/jobs/${jobId}/candidates?${qs}` : `/api/jobs/${jobId}/candidates`;
    return request<CandidateListResponse>(endpoint, { method: 'GET' });
  },

  async getAllCandidates(params?: {
    jobId?: number;
    search?: string;
    status?: string;
    page?: number;
    pageSize?: number;
  }): Promise<CandidateListResponse> {
    const query = new URLSearchParams();
    if (params?.jobId !== undefined && params.jobId !== null) {
      query.append('job_id', String(params.jobId));
    }
    if (params?.search && params.search.trim()) {
      query.append('search', params.search.trim());
    }
    if (params?.status && params.status !== 'all') {
      query.append('status', params.status);
    }
    if (params?.page) {
      query.append('page', String(params.page));
    }
    if (params?.pageSize) {
      query.append('page_size', String(params.pageSize));
    }

    const qs = query.toString();
    const endpoint = qs ? `/api/candidates?${qs}` : '/api/candidates';
    return request<CandidateListResponse>(endpoint, { method: 'GET' });
  },

  async getCandidateDetails(candidateId: number): Promise<Candidate> {
    return request<Candidate>(`/api/candidates/${candidateId}`, { method: 'GET' });
  },

  async getCandidate(candidateId: number): Promise<Candidate> {
    return this.getCandidateDetails(candidateId);
  },

  async deleteCandidate(candidateId: number): Promise<DeleteCandidateResponse> {
    return request<DeleteCandidateResponse>(`/api/candidates/${candidateId}`, { method: 'DELETE' });
  },

  async retryExtraction(candidateId: number): Promise<Candidate> {
    return request<Candidate>(`/api/candidates/${candidateId}/retry-extraction`, { method: 'POST' });
  },
};
