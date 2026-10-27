export type ProcessingStatus = 'processing' | 'completed' | 'failed';

export interface Candidate {
  id: number;
  job_id: number;
  recruiter_id: number;
  original_filename: string;
  file_type: string;
  file_size: number;
  processing_status: ProcessingStatus;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  job_title?: string | null;
  extracted_text?: string | null;
  relevance_score?: number | null;
  fit_category?: 'No Fit' | 'Potential Fit' | 'Good Fit' | string | null;
  ranking_position?: number | null;
  predicted_probabilities?: string | null;
  model_version?: string | null;
  full_name?: string | null;
  email?: string | null;
  phone?: string | null;
  location?: string | null;
  linkedin_url?: string | null;
  github_url?: string | null;
  portfolio_url?: string | null;
  matched_skills?: string | null;
  structured_resume?: string | null;
  explanation?: string | null;
}

export interface SkillEvidenceItem {
  skill_name: string;
  status: 'Matched' | 'Partial' | 'Not evidenced' | 'Parser limitation' | string;
  confidence: number;
  evidence_text?: string | null;
  section_found?: string | null;
  reason?: string | null;
}

export interface ExperienceHighlightItem {
  role_or_project: string;
  organization?: string | null;
  duration_or_date?: string | null;
  relevance_summary: string;
  technologies_evidenced: string[];
  is_internship: boolean;
}

export interface ScoreBreakdownItem {
  semantic_similarity_score: number;
  required_skill_coverage: number;
  preferred_skill_coverage: number;
  experience_project_score: number;
  final_relevance_score: number;
  formula_description: string;
}

export interface CandidateExplanationData {
  rank?: number | null;
  relevance_score: number;
  fit_category: string;
  overall_summary: string;
  strongest_match: string;
  top_missing_requirement?: string | null;
  required_skills: SkillEvidenceItem[];
  preferred_skills: SkillEvidenceItem[];
  experience_highlights: ExperienceHighlightItem[];
  peer_comparison?: string | null;
  score_breakdown: ScoreBreakdownItem;
}

export interface RankJobResponse {
  job_id: number;
  job_title: string;
  model_used: string;
  scoring_note: string;
  total_candidates_ranked: number;
  ranked_candidates: Candidate[];
  ranked_at: string;
}

export interface CandidateListResponse {
  items: Candidate[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface DuplicateUploadItem {
  filename: string;
  file_size: number;
  message: string;
  existing_candidate_id: number;
}

export interface UploadErrorItem {
  filename: string;
  reason: string;
}

export interface BatchUploadResponse {
  uploaded: Candidate[];
  duplicates: DuplicateUploadItem[];
  errors: UploadErrorItem[];
  total_uploaded: number;
  total_duplicates: number;
  total_errors: number;
}

export interface DeleteCandidateResponse {
  message: string;
  id: number;
}
