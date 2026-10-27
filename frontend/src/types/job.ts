export type JobStatus = 'draft' | 'active' | 'archived';

export type EmploymentType =
  | 'Full-time'
  | 'Part-time'
  | 'Contract'
  | 'Internship'
  | 'Freelance';

export type WorkplaceType = 'On-site' | 'Hybrid' | 'Remote';

export interface Job {
  id: number;
  recruiter_id: number;
  title: string;
  department?: string | null;
  location?: string | null;
  employment_type: EmploymentType | string;
  workplace_type: WorkplaceType | string;
  description: string;
  required_skills: string[];
  preferred_skills: string[];
  minimum_experience?: number | null;
  status: JobStatus;
  created_at: string;
  updated_at: string;
}

export interface JobCreateInput {
  title: string;
  department?: string | null;
  location?: string | null;
  employment_type?: EmploymentType | string;
  workplace_type?: WorkplaceType | string;
  description: string;
  required_skills?: string[];
  preferred_skills?: string[];
  minimum_experience?: number | null;
  status?: JobStatus;
}

export interface JobUpdateInput {
  title?: string;
  department?: string | null;
  location?: string | null;
  employment_type?: EmploymentType | string;
  workplace_type?: WorkplaceType | string;
  description?: string;
  required_skills?: string[];
  preferred_skills?: string[];
  minimum_experience?: number | null;
  status?: JobStatus;
}

export interface DeleteJobResponse {
  message: string;
  id: number;
}
