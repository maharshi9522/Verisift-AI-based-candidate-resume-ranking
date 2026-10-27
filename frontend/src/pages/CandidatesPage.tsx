import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { CandidateCard } from '../components/candidates/CandidateCard';
import { CandidateDetailModal } from '../components/candidates/CandidateDetailModal';
import { ResumeUploadModal } from '../components/candidates/ResumeUploadModal';
import { DeleteConfirmModal } from '../components/jobs/DeleteConfirmModal';
import { candidateService } from '../services/candidates';
import { jobService } from '../services/jobs';
import type { Candidate, BatchUploadResponse } from '../types/candidate';
import type { Job } from '../types/job';
import {
  UploadCloud,
  Search,
  RotateCcw,
  Users,
  Briefcase,
  AlertCircle,
  Loader2,
  CheckCircle2,
  X,
  ChevronLeft,
  ChevronRight,
  ArrowRight,
  Sparkles,
  Download,
  FileSpreadsheet,
  FileText,
} from 'lucide-react';

export const CandidatesPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlJobId = searchParams.get('jobId') ? Number(searchParams.get('jobId')) : null;

  // Data states
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(urlJobId);
  const [totalCandidates, setTotalCandidates] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  // UI / Modal states
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadTargetJob, setUploadTargetJob] = useState<Job | null>(null);

  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [candidateToDelete, setCandidateToDelete] = useState<Candidate | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [retryingId, setRetryingId] = useState<number | null>(null);
  const [isRanking, setIsRanking] = useState(false);

  // Toast feedback
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const showToast = (message: string) => {
    setToastMessage(message);
    setTimeout(() => {
      setToastMessage((prev) => (prev === message ? null : prev));
    }, 4000);
  };

  const handleRankCandidates = async () => {
    if (!selectedJobId) {
      showToast('Please select a specific job role to rank candidates.');
      return;
    }
    try {
      setIsRanking(true);
      const res = await candidateService.rankJobCandidates(selectedJobId);
      showToast(`Ranked ${res.total_candidates_ranked} candidate${res.total_candidates_ranked === 1 ? '' : 's'} using deep semantic matching!`);
      fetchCandidates();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Ranking failed. Ensure candidates have completed extraction.';
      showToast(msg);
    } finally {
      setIsRanking(false);
    }
  };

  // Export Shortlist state
  const [exportCountType, setExportCountType] = useState<'100' | '200' | 'custom'>('100');
  const [customExportCount, setCustomExportCount] = useState<string>('50');
  const [exportFormat, setExportFormat] = useState<'xlsx' | 'pdf'>('xlsx');
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  // Validate custom count
  const parsedCustom = parseInt(customExportCount, 10);
  const isCustomValid = !isNaN(parsedCustom) && parsedCustom > 0 && parsedCustom <= 1000 && Number.isInteger(Number(customExportCount));
  const effectiveLimit = exportCountType === 'custom' ? (isCustomValid ? parsedCustom : null) : parseInt(exportCountType, 10);

  const handleDownloadShortlist = async () => {
    if (!selectedJobId) {
      showToast('Please select a specific job role to download its shortlist.');
      return;
    }
    if (effectiveLimit === null || effectiveLimit <= 0) {
      setExportError('Please enter a valid positive integer between 1 and 1000.');
      return;
    }
    try {
      setIsExporting(true);
      setExportError(null);
      await candidateService.exportCandidateShortlist(selectedJobId, effectiveLimit, exportFormat);
      showToast(`Shortlist (${exportFormat.toUpperCase()}) downloaded successfully!`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Export failed. Please try again.';
      setExportError(msg);
      showToast(msg);
    } finally {
      setIsExporting(false);
    }
  };

  // Sync selectedJobId if URL changes
  useEffect(() => {
    if (urlJobId !== selectedJobId) {
      setSelectedJobId(urlJobId);
      setCurrentPage(1);
    }
  }, [urlJobId, selectedJobId]);

  // Load recruiter jobs for dropdown
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const jobsData = await jobService.getJobs();
        setJobs(jobsData);
      } catch {
        // Handle silently
      }
    };
    fetchJobs();
  }, []);

  // Fetch candidates from API
  const fetchCandidates = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await candidateService.getAllCandidates({
        jobId: selectedJobId || undefined,
        search: searchQuery.trim() || undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        page: currentPage,
        pageSize: 20,
      });
      setCandidates(data.items);
      setTotalCandidates(data.total);
      setTotalPages(data.total_pages);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load candidates.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [selectedJobId, searchQuery, statusFilter, currentPage]);

  useEffect(() => {
    fetchCandidates();
  }, [fetchCandidates]);

  // Handle job filter change
  const handleJobFilterChange = (newJobIdStr: string) => {
    const newId = newJobIdStr === 'all' ? null : Number(newJobIdStr);
    setSelectedJobId(newId);
    setCurrentPage(1);
    if (newId) {
      setSearchParams({ jobId: String(newId) });
    } else {
      setSearchParams({});
    }
  };

  // Open upload modal
  const handleOpenUpload = () => {
    if (jobs.length === 0) {
      showToast('Please create a job role before uploading candidate resumes.');
      return;
    }

    if (selectedJobId) {
      const target = jobs.find((j) => j.id === selectedJobId) || jobs[0];
      setUploadTargetJob(target);
    } else {
      // Pick first active or available role
      const target = jobs.find((j) => j.status === 'active') || jobs[0];
      setUploadTargetJob(target);
    }
    setIsUploadModalOpen(true);
  };

  const handleUploadSuccess = (result: BatchUploadResponse) => {
    fetchCandidates();
    showToast(
      `Ingested ${result.total_uploaded} resume${result.total_uploaded === 1 ? '' : 's'}${
        result.total_duplicates > 0 ? ` (${result.total_duplicates} duplicate skipped)` : ''
      }.`
    );
  };

  // View candidate and ensure full extracted text is loaded
  const handleViewCandidate = async (candidate: Candidate) => {
    setSelectedCandidate(candidate);
    try {
      const full = await candidateService.getCandidate(candidate.id);
      setSelectedCandidate(full);
    } catch {
      // keep candidate already set in state
    }
  };

  // Retry text extraction
  const handleRetryExtraction = async (candidate: Candidate) => {
    setRetryingId(candidate.id);
    try {
      const updated = await candidateService.retryExtraction(candidate.id);
      setCandidates((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
      if (selectedCandidate?.id === candidate.id) {
        setSelectedCandidate(updated);
      }
      showToast(`Extraction retried for ${candidate.original_filename}.`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Retry failed.';
      showToast(msg);
    } finally {
      setRetryingId(null);
    }
  };

  // Delete candidate
  const handleDeleteConfirm = async () => {
    if (!candidateToDelete) return;
    setIsDeleting(true);
    try {
      await candidateService.deleteCandidate(candidateToDelete.id);
      setCandidates((prev) => prev.filter((c) => c.id !== candidateToDelete.id));
      setTotalCandidates((prev) => Math.max(0, prev - 1));
      if (selectedCandidate?.id === candidateToDelete.id) {
        setSelectedCandidate(null);
      }
      showToast(`Removed candidate "${candidateToDelete.original_filename}".`);
      setCandidateToDelete(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to delete candidate.';
      showToast(msg);
    } finally {
      setIsDeleting(false);
    }
  };

  const isFilterActive = searchQuery.trim() !== '' || statusFilter !== 'all' || selectedJobId !== null;
  const currentJob = jobs.find((j) => j.id === selectedJobId);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Toast Notification */}
      {toastMessage && (
        <div
          style={{
            position: 'fixed',
            bottom: '24px',
            right: '24px',
            backgroundColor: '#12343B',
            color: '#FFFFFF',
            padding: '12px 18px',
            borderRadius: 'var(--radius-sm)',
            boxShadow: 'var(--shadow-modal)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            zIndex: 1000,
            fontSize: '13px',
            fontWeight: 500,
            border: '1px solid rgba(39, 211, 176, 0.4)',
          }}
          className="animate-slide-up"
        >
          <CheckCircle2 size={16} color="#27D3B0" />
          <span>{toastMessage}</span>
          <button
            onClick={() => setToastMessage(null)}
            style={{
              background: 'none',
              border: 'none',
              color: '#8FA7AD',
              cursor: 'pointer',
              padding: '2px',
              marginLeft: '6px',
            }}
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1
              style={{
                fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                fontSize: '24px',
                fontWeight: 600,
                color: 'var(--color-text)',
                letterSpacing: '-0.02em',
                marginBottom: '2px',
              }}
            >
              Candidates
            </h1>
            {currentJob && (
              <span
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: 'var(--color-primary)',
                  backgroundColor: 'rgba(39, 211, 176, 0.12)',
                  border: '1px solid rgba(39, 211, 176, 0.3)',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-full)',
                }}
              >
                {currentJob.title}
              </span>
            )}
          </div>
          <p style={{ fontSize: '13.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
            Uploaded applicant resumes, text extraction status, and document inspection.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {selectedJobId && (
            <Button
              variant="secondary"
              size="md"
              leftIcon={
                isRanking ? (
                  <Loader2 size={15} className="animate-spin" />
                ) : (
                  <Sparkles size={15} color="#059669" />
                )
              }
              onClick={handleRankCandidates}
              disabled={isRanking}
            >
              {isRanking ? 'Ranking with AI...' : 'Rank Candidates with AI'}
            </Button>
          )}
          <Button
            variant="primary"
            size="md"
            leftIcon={<UploadCloud size={15} />}
            onClick={handleOpenUpload}
          >
            Upload resumes
          </Button>
        </div>
      </div>

      {currentJob && (
        <div
          style={{
            backgroundColor: '#F8FAFC',
            border: '1px solid #E2E8F0',
            borderRadius: '6px',
            padding: '8px 14px',
            fontSize: '12.5px',
            color: '#475569',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <Sparkles size={14} color="#059669" />
          <span>
            <strong>AI Matching Engine:</strong> Scores represent an <em>ordinal relevance score (0–100)</em> predicted by the deep pairwise matching MLP, sorted in descending rank order.
          </span>
        </div>
      )}

      {currentJob && (
        <div
          style={{
            backgroundColor: '#FFFFFF',
            border: '1px solid #E2E8F0',
            borderRadius: '8px',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '12px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          }}
        >
          {/* Left: Title & description */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '6px',
                backgroundColor: '#EFF6FF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#2563EB',
              }}
            >
              <Download size={16} />
            </div>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#1E293B' }}>
                Download Ranked Shortlist
              </div>
              <div style={{ fontSize: '11.5px', color: '#64748B' }}>
                Export candidate rankings for {currentJob.title}
                {totalCandidates > 0 && ` (${totalCandidates} candidates)`}
              </div>
            </div>
          </div>

          {/* Right: Controls & Download button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            {/* Candidate count dropdown */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label htmlFor="export-count-select" style={{ fontSize: '12px', color: '#64748B', fontWeight: 500 }}>
                Count:
              </label>
              <select
                id="export-count-select"
                value={exportCountType}
                onChange={(e) => {
                  setExportCountType(e.target.value as any);
                  setExportError(null);
                }}
                className="form-input"
                style={{
                  height: '32px',
                  fontSize: '12px',
                  padding: '0 8px',
                  width: '110px',
                  cursor: 'pointer',
                  borderRadius: '6px',
                  border: '1px solid #CBD5E1',
                }}
              >
                <option value="100">Top 100</option>
                <option value="200">Top 200</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            {/* Custom count input if custom is selected */}
            {exportCountType === 'custom' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <input
                  id="custom-export-count"
                  type="number"
                  min="1"
                  max="1000"
                  step="1"
                  placeholder="Count"
                  value={customExportCount}
                  onChange={(e) => {
                    setCustomExportCount(e.target.value);
                    setExportError(null);
                  }}
                  style={{
                    height: '32px',
                    width: '75px',
                    fontSize: '12px',
                    padding: '0 8px',
                    borderRadius: '6px',
                    border: !isCustomValid ? '1px solid #EF4444' : '1px solid #CBD5E1',
                    outline: 'none',
                  }}
                />
              </div>
            )}

            {/* Format selector (Excel or PDF) */}
            <div
              style={{
                display: 'flex',
                backgroundColor: '#F1F5F9',
                borderRadius: '6px',
                padding: '2px',
                gap: '2px',
                border: '1px solid #E2E8F0',
              }}
            >
              <button
                type="button"
                id="export-format-xlsx"
                onClick={() => setExportFormat('xlsx')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  border: 'none',
                  padding: '4px 10px',
                  fontSize: '11.5px',
                  fontWeight: exportFormat === 'xlsx' ? 600 : 500,
                  color: exportFormat === 'xlsx' ? '#1E293B' : '#64748B',
                  backgroundColor: exportFormat === 'xlsx' ? '#FFFFFF' : 'transparent',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  boxShadow: exportFormat === 'xlsx' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
                }}
              >
                <FileSpreadsheet size={13} color="#16A34A" />
                Excel (.xlsx)
              </button>
              <button
                type="button"
                id="export-format-pdf"
                onClick={() => setExportFormat('pdf')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  border: 'none',
                  padding: '4px 10px',
                  fontSize: '11.5px',
                  fontWeight: exportFormat === 'pdf' ? 600 : 500,
                  color: exportFormat === 'pdf' ? '#1E293B' : '#64748B',
                  backgroundColor: exportFormat === 'pdf' ? '#FFFFFF' : 'transparent',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  boxShadow: exportFormat === 'pdf' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
                }}
              >
                <FileText size={13} color="#DC2626" />
                PDF
              </button>
            </div>

            {/* Download Button */}
            <Button
              id="download-shortlist-btn"
              variant="primary"
              size="sm"
              leftIcon={
                isExporting ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Download size={13} />
                )
              }
              onClick={handleDownloadShortlist}
              disabled={isExporting || totalCandidates === 0 || effectiveLimit === null || effectiveLimit <= 0}
            >
              {isExporting ? 'Exporting...' : 'Download Shortlist'}
            </Button>
          </div>

          {/* Validation or Error Message */}
          {exportError && (
            <div
              style={{
                width: '100%',
                fontSize: '11.5px',
                color: '#DC2626',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                marginTop: '4px',
              }}
            >
              <AlertCircle size={13} />
              <span>{exportError}</span>
            </div>
          )}

          {/* Notice when requested count exceeds available count */}
          {effectiveLimit !== null && effectiveLimit > totalCandidates && totalCandidates > 0 && (
            <div
              style={{
                width: '100%',
                fontSize: '11.5px',
                color: '#64748B',
                fontStyle: 'italic',
              }}
            >
              Note: Requested Top {effectiveLimit}, but only {totalCandidates} candidate{totalCandidates === 1 ? '' : 's'} available. All {totalCandidates} will be included.
            </div>
          )}
        </div>
      )}

      {/* Filter and Search Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          flexWrap: 'wrap',
        }}
      >
        {/* Left: Job selector & Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          {/* Job Dropdown Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Briefcase size={14} color="#61777D" />
            <select
              value={selectedJobId ? String(selectedJobId) : 'all'}
              onChange={(e) => handleJobFilterChange(e.target.value)}
              className="form-input"
              style={{
                height: '36px',
                fontSize: '12.5px',
                padding: '0 10px',
                width: '210px',
                cursor: 'pointer',
              }}
            >
              <option value="all">All Open Roles</option>
              {jobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {job.title}
                </option>
              ))}
            </select>
          </div>

          {/* Search by filename */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-xs)',
              padding: '0 12px',
              height: '36px',
              width: '260px',
            }}
          >
            <Search size={14} color="#8E9FA4" />
            <input
              type="text"
              placeholder="Search by filename..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              style={{
                border: 'none',
                outline: 'none',
                fontSize: '13px',
                backgroundColor: 'transparent',
                color: 'var(--color-text)',
                width: '100%',
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--color-text-secondary)',
                  cursor: 'pointer',
                  padding: '2px',
                }}
                aria-label="Clear search"
              >
                <X size={13} />
              </button>
            )}
          </div>
        </div>

        {/* Right: Status Filter Tabs */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-xs)',
            padding: '2px',
            gap: '2px',
          }}
        >
          {(['all', 'completed', 'failed', 'processing'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => {
                setStatusFilter(tab);
                setCurrentPage(1);
              }}
              style={{
                border: 'none',
                background: statusFilter === tab ? 'var(--color-hover)' : 'transparent',
                color: statusFilter === tab ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                fontWeight: statusFilter === tab ? 600 : 500,
                fontSize: '12.5px',
                padding: '5px 12px',
                borderRadius: 'var(--radius-xs)',
                cursor: 'pointer',
                textTransform: 'capitalize',
                transition: 'all var(--transition-fast)',
              }}
            >
              {tab === 'all' ? 'All Status' : tab}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      {isLoading ? (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '60px 20px',
            gap: '12px',
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <Loader2 size={24} color="var(--color-primary)" style={{ animation: 'spin 1s linear infinite' }} />
          <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>Loading candidates...</span>
        </div>
      ) : error ? (
        <div
          style={{
            padding: '24px',
            backgroundColor: 'var(--color-error-bg)',
            border: '1px solid rgba(195, 68, 68, 0.25)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--color-error)' }}>
            <AlertCircle size={18} />
            <span style={{ fontSize: '13.5px' }}>{error}</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => fetchCandidates()}>
            Retry
          </Button>
        </div>
      ) : candidates.length === 0 ? (
        isFilterActive ? (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '48px 24px',
              textAlign: 'center',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--shadow-card)',
            }}
            className="animate-fade-in"
          >
            <div
              style={{
                width: '42px',
                height: '42px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'var(--color-hover)',
                color: 'var(--color-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
                border: '1px solid var(--color-border)',
              }}
            >
              <Search size={20} />
            </div>

            <h3
              style={{
                fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                fontSize: '17px',
                fontWeight: 600,
                color: 'var(--color-text)',
                marginBottom: '6px',
              }}
            >
              No matching candidates
            </h3>

            <p
              style={{
                fontSize: '13.5px',
                color: 'var(--color-text-secondary)',
                maxWidth: '380px',
                lineHeight: 1.5,
                marginBottom: '18px',
              }}
            >
              No candidate files matched your search or status filter. Try clearing filters to view all records.
            </p>

            <Button
              variant="secondary"
              size="md"
              leftIcon={<RotateCcw size={14} />}
              onClick={() => {
                setSearchQuery('');
                setStatusFilter('all');
                setSelectedJobId(null);
                setSearchParams({});
              }}
            >
              Clear search & filters
            </Button>
          </div>
        ) : (
          /* True Zero-State */
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '56px 24px',
              textAlign: 'center',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--shadow-card)',
            }}
            className="animate-fade-in"
          >
            <div
              style={{
                width: '44px',
                height: '44px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'var(--color-hover)',
                color: 'var(--color-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px',
              }}
            >
              <Users size={22} />
            </div>

            <h3
              style={{
                fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                fontSize: '17px',
                fontWeight: 600,
                color: 'var(--color-text)',
                marginBottom: '6px',
              }}
            >
              No candidates yet
            </h3>

            <p
              style={{
                fontSize: '13.5px',
                color: 'var(--color-text-secondary)',
                maxWidth: '380px',
                lineHeight: 1.5,
                marginBottom: '20px',
              }}
            >
              Upload PDF or DOCX candidate resumes to extract text content and review applicant profiles.
            </p>

            {jobs.length > 0 ? (
              <Button
                variant="primary"
                size="md"
                leftIcon={<UploadCloud size={15} />}
                onClick={handleOpenUpload}
              >
                Upload resumes
              </Button>
            ) : (
              <Link to="/jobs">
                <Button variant="primary" size="md" rightIcon={<ArrowRight size={14} />}>
                  Create your first role
                </Button>
              </Link>
            )}
          </div>
        )
      ) : (
        /* Candidates List */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {candidates.map((candidate) => (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              onClick={(c) => handleViewCandidate(c)}
              onDelete={(c) => setCandidateToDelete(c)}
              onRetry={(c) => handleRetryExtraction(c)}
              isRetrying={retryingId === candidate.id}
            />
          ))}

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 6px',
                fontSize: '12.5px',
                color: 'var(--color-text-secondary)',
              }}
            >
              <span>
                Showing {candidates.length} of {totalCandidates} candidates
              </span>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage <= 1}
                  leftIcon={<ChevronLeft size={14} />}
                >
                  Previous
                </Button>
                <span style={{ padding: '0 8px', fontWeight: 600 }}>
                  {currentPage} of {totalPages}
                </span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage >= totalPages}
                  rightIcon={<ChevronRight size={14} />}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload Resumes Modal */}
      {uploadTargetJob && (
        <ResumeUploadModal
          isOpen={isUploadModalOpen}
          onClose={() => {
            setIsUploadModalOpen(false);
            setUploadTargetJob(null);
          }}
          jobId={uploadTargetJob.id}
          jobTitle={uploadTargetJob.title}
          onUploadSuccess={handleUploadSuccess}
        />
      )}

      {/* Candidate Detail Modal */}
      <CandidateDetailModal
        candidate={selectedCandidate}
        isOpen={!!selectedCandidate}
        onClose={() => setSelectedCandidate(null)}
        onRetry={handleRetryExtraction}
        onDelete={(c) => {
          setSelectedCandidate(null);
          setCandidateToDelete(c);
        }}
        isRetrying={retryingId === selectedCandidate?.id}
      />

      {/* Safe Delete Confirmation */}
      <DeleteConfirmModal
        isOpen={!!candidateToDelete}
        onClose={() => setCandidateToDelete(null)}
        onConfirm={handleDeleteConfirm}
        jobTitle={candidateToDelete?.original_filename || ''}
        isDeleting={isDeleting}
      />
    </div>
  );
};
