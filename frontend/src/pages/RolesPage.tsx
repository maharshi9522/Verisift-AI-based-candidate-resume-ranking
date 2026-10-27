import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '../components/ui/Button';
import { RolesEmptyState } from '../components/dashboard/RolesEmptyState';
import { JobFormModal } from '../components/jobs/JobFormModal';
import { JobDetailModal } from '../components/jobs/JobDetailModal';
import { DeleteConfirmModal } from '../components/jobs/DeleteConfirmModal';
import { JobCard } from '../components/jobs/JobCard';
import type { Job, JobStatus } from '../types/job';
import { jobService } from '../services/jobs';
import { Plus, Search, Loader2, AlertCircle, CheckCircle2, X } from 'lucide-react';

export const RolesPage: React.FC = () => {
  // Data state
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Modals state
  const [isFormModalOpen, setIsFormModalOpen] = useState(false);
  const [jobToEdit, setJobToEdit] = useState<Job | null>(null);

  const [selectedJobForDetail, setSelectedJobForDetail] = useState<Job | null>(null);

  const [jobToDelete, setJobToDelete] = useState<Job | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Toast notification
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const showToast = (message: string) => {
    setToastMessage(message);
    setTimeout(() => {
      setToastMessage((prev) => (prev === message ? null : prev));
    }, 4000);
  };

  // Fetch jobs from backend
  const fetchJobs = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await jobService.getJobs({
        status: statusFilter !== 'all' ? statusFilter : undefined,
        search: searchQuery.trim() || undefined,
      });
      setJobs(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to load jobs. Please check your network and try again.');
      }
    } finally {
      setIsLoading(false);
    }
  }, [statusFilter, searchQuery]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Handlers
  const handleOpenCreate = () => {
    setJobToEdit(null);
    setIsFormModalOpen(true);
  };

  const handleOpenEdit = (job: Job) => {
    setJobToEdit(job);
    setIsFormModalOpen(true);
  };

  const handleFormSuccess = (job: Job, isEdit: boolean) => {
    if (isEdit) {
      setJobs((prev) => prev.map((j) => (j.id === job.id ? job : j)));
      if (selectedJobForDetail?.id === job.id) {
        setSelectedJobForDetail(job);
      }
      showToast(`Role "${job.title}" updated successfully.`);
    } else {
      setJobs((prev) => [job, ...prev]);
      showToast(`Role "${job.title}" created successfully.`);
    }
  };

  const handleToggleStatus = async (job: Job) => {
    const nextStatus: JobStatus = job.status === 'active' ? 'draft' : 'active';
    try {
      const updated = await jobService.updateJob(job.id, { status: nextStatus });
      setJobs((prev) => prev.map((j) => (j.id === updated.id ? updated : j)));
      if (selectedJobForDetail?.id === job.id) {
        setSelectedJobForDetail(updated);
      }
      showToast(
        nextStatus === 'active'
          ? `Role "${job.title}" is now active.`
          : `Role "${job.title}" moved to draft.`
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to update status.';
      showToast(msg);
    }
  };

  const handleArchive = async (job: Job) => {
    try {
      const updated = await jobService.archiveJob(job.id);
      setJobs((prev) => prev.map((j) => (j.id === updated.id ? updated : j)));
      if (selectedJobForDetail?.id === job.id) {
        setSelectedJobForDetail(updated);
      }
      showToast(`Role "${job.title}" archived.`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to archive role.';
      showToast(msg);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!jobToDelete) return;
    setIsDeleting(true);
    try {
      await jobService.deleteJob(jobToDelete.id);
      setJobs((prev) => prev.filter((j) => j.id !== jobToDelete.id));
      if (selectedJobForDetail?.id === jobToDelete.id) {
        setSelectedJobForDetail(null);
      }
      showToast(`Role "${jobToDelete.title}" deleted.`);
      setJobToDelete(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to delete role.';
      showToast(msg);
    } finally {
      setIsDeleting(false);
    }
  };

  const isFilterActive = searchQuery.trim() !== '' || statusFilter !== 'all';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Toast Feedback */}
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
            Jobs
          </h1>
          <p style={{ fontSize: '13.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
            Open recruitment positions and target competency criteria.
          </p>
        </div>

        <Button
          variant="primary"
          size="md"
          leftIcon={<Plus size={15} />}
          onClick={handleOpenCreate}
        >
          Create role
        </Button>
      </div>

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
        {/* Search Input */}
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
            width: '280px',
          }}
        >
          <Search size={14} color="#8E9FA4" />
          <input
            type="text"
            placeholder="Search by title, dept, location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
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

        {/* Status Filter Tabs */}
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
          {(['all', 'active', 'draft', 'archived'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setStatusFilter(tab)}
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
              {tab === 'all' ? 'All Roles' : tab}
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
          <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>Loading roles...</span>
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
          <Button variant="secondary" size="sm" onClick={() => fetchJobs()}>
            Retry
          </Button>
        </div>
      ) : jobs.length === 0 ? (
        <RolesEmptyState
          onCreateRole={handleOpenCreate}
          isFiltered={isFilterActive}
          onClearFilters={() => {
            setSearchQuery('');
            setStatusFilter('all');
          }}
        />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {jobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              onClick={(j) => setSelectedJobForDetail(j)}
              onEdit={(j) => handleOpenEdit(j)}
              onDelete={(j) => setJobToDelete(j)}
              onArchive={(j) => handleArchive(j)}
              onToggleStatus={(j) => handleToggleStatus(j)}
            />
          ))}
        </div>
      )}

      {/* Create / Edit Role Modal */}
      <JobFormModal
        isOpen={isFormModalOpen}
        onClose={() => setIsFormModalOpen(false)}
        onSuccess={handleFormSuccess}
        jobToEdit={jobToEdit}
      />

      {/* Detailed Role Inspection View */}
      <JobDetailModal
        job={selectedJobForDetail}
        isOpen={!!selectedJobForDetail}
        onClose={() => setSelectedJobForDetail(null)}
        onEdit={(j) => {
          setSelectedJobForDetail(null);
          handleOpenEdit(j);
        }}
        onArchive={(j) => handleArchive(j)}
        onDelete={(j) => {
          setSelectedJobForDetail(null);
          setJobToDelete(j);
        }}
        onToggleStatus={(j) => handleToggleStatus(j)}
      />

      {/* Safe Delete Confirmation */}
      <DeleteConfirmModal
        isOpen={!!jobToDelete}
        onClose={() => setJobToDelete(null)}
        onConfirm={handleDeleteConfirm}
        jobTitle={jobToDelete?.title || ''}
        isDeleting={isDeleting}
      />
    </div>
  );
};
