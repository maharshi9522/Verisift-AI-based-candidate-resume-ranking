import { request } from './api';
import type { Job, JobCreateInput, JobUpdateInput, DeleteJobResponse } from '../types/job';

export const jobService = {
  async getJobs(params?: { status?: string; search?: string }): Promise<Job[]> {
    const query = new URLSearchParams();
    if (params?.status && params.status !== 'all') {
      query.append('status', params.status);
    }
    if (params?.search && params.search.trim()) {
      query.append('search', params.search.trim());
    }

    const queryString = query.toString();
    const endpoint = queryString ? `/api/jobs?${queryString}` : '/api/jobs';
    return request<Job[]>(endpoint, {
      method: 'GET',
    });
  },

  async getJobById(id: number): Promise<Job> {
    return request<Job>(`/api/jobs/${id}`, {
      method: 'GET',
    });
  },

  async createJob(payload: JobCreateInput): Promise<Job> {
    return request<Job>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async updateJob(id: number, payload: JobUpdateInput): Promise<Job> {
    return request<Job>(`/api/jobs/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
  },

  async deleteJob(id: number): Promise<DeleteJobResponse> {
    return request<DeleteJobResponse>(`/api/jobs/${id}`, {
      method: 'DELETE',
    });
  },

  async archiveJob(id: number): Promise<Job> {
    return request<Job>(`/api/jobs/${id}/archive`, {
      method: 'POST',
    });
  },
};
