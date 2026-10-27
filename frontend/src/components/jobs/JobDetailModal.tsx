import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import type { Job } from '../../types/job';
import {
  Briefcase,
  MapPin,
  Clock,
  Laptop,
  Calendar,
  Edit3,
  Archive,
  Trash2,
  CheckCircle,
  FileText,
  Users,
} from 'lucide-react';

interface JobDetailModalProps {
  job: Job | null;
  isOpen: boolean;
  onClose: () => void;
  onEdit: (job: Job) => void;
  onArchive: (job: Job) => void;
  onDelete: (job: Job) => void;
  onToggleStatus: (job: Job) => void;
}

export const JobDetailModal: React.FC<JobDetailModalProps> = ({
  job,
  isOpen,
  onClose,
  onEdit,
  onArchive,
  onDelete,
  onToggleStatus,
}) => {
  const navigate = useNavigate();
  if (!job) return null;

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return isoString;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge variant="turquoise">Active Role</Badge>;
      case 'draft':
        return <Badge variant="muted">Draft</Badge>;
      case 'archived':
        return <Badge variant="warning">Archived</Badge>;
      default:
        return <Badge variant="teal">{status}</Badge>;
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={job.title}
      description={job.department ? `${job.department} Department` : undefined}
      maxWidth="720px"
      footer={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDelete(job)}
            leftIcon={<Trash2 size={14} color="var(--color-error)" />}
            style={{ color: 'var(--color-error)' }}
          >
            Delete role
          </Button>

          <div style={{ display: 'flex', gap: '8px' }}>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                onClose();
                navigate(`/candidates?jobId=${job.id}`);
              }}
              leftIcon={<Users size={14} />}
            >
              Candidates
            </Button>

            {job.status !== 'archived' && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onArchive(job)}
                leftIcon={<Archive size={14} />}
              >
                Archive
              </Button>
            )}

            <Button
              variant="secondary"
              size="sm"
              onClick={() => onToggleStatus(job)}
              leftIcon={<CheckCircle size={14} />}
            >
              {job.status === 'active' ? 'Move to Draft' : 'Publish as Active'}
            </Button>

            <Button
              variant="primary"
              size="sm"
              onClick={() => {
                onClose();
                onEdit(job);
              }}
              leftIcon={<Edit3 size={14} />}
            >
              Edit role
            </Button>
          </div>
        </div>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Status & Key Attributes Header */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '12px',
            paddingBottom: '16px',
            borderBottom: '1px solid var(--color-border)',
          }}
        >
          {getStatusBadge(job.status)}

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text-secondary)' }}>
            <Briefcase size={14} color="#61777D" />
            <span>{job.employment_type}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text-secondary)' }}>
            <Laptop size={14} color="#61777D" />
            <span>{job.workplace_type}</span>
          </div>

          {job.location && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text-secondary)' }}>
              <MapPin size={14} color="#61777D" />
              <span>{job.location}</span>
            </div>
          )}

          {job.minimum_experience !== null && job.minimum_experience !== undefined && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text-secondary)' }}>
              <Clock size={14} color="#61777D" />
              <span>{job.minimum_experience}+ yrs exp</span>
            </div>
          )}

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              color: 'var(--color-text-muted)',
              marginLeft: 'auto',
            }}
          >
            <Calendar size={13} />
            <span>Created {formatDate(job.created_at)}</span>
          </div>
        </div>

        {/* Required Competencies */}
        <div>
          <h4
            style={{
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--color-text)',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            Required Competencies
          </h4>
          {job.required_skills && job.required_skills.length > 0 ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {job.required_skills.map((skill) => (
                <span
                  key={skill}
                  style={{
                    padding: '3px 10px',
                    backgroundColor: 'rgba(39, 211, 176, 0.12)',
                    color: '#12343B',
                    border: '1px solid rgba(39, 211, 176, 0.35)',
                    borderRadius: 'var(--radius-xs)',
                    fontSize: '12px',
                    fontWeight: 600,
                  }}
                >
                  {skill}
                </span>
              ))}
            </div>
          ) : (
            <span style={{ fontSize: '13px', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
              No specific required competencies listed.
            </span>
          )}
        </div>

        {/* Preferred Competencies */}
        {job.preferred_skills && job.preferred_skills.length > 0 && (
          <div>
            <h4
              style={{
                fontSize: '13px',
                fontWeight: 600,
                color: 'var(--color-text)',
                marginBottom: '8px',
              }}
            >
              Preferred Competencies
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {job.preferred_skills.map((skill) => (
                <span
                  key={skill}
                  style={{
                    padding: '3px 10px',
                    backgroundColor: 'var(--color-hover)',
                    color: 'var(--color-text)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-xs)',
                    fontSize: '12px',
                    fontWeight: 500,
                  }}
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Job Description */}
        <div>
          <h4
            style={{
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--color-text)',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <FileText size={15} color="var(--color-primary)" />
            Role Description & Responsibilities
          </h4>
          <div
            style={{
              backgroundColor: 'var(--color-background)',
              padding: '16px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--color-border)',
              fontSize: '13.5px',
              color: 'var(--color-text)',
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap',
            }}
          >
            {job.description}
          </div>
        </div>
      </div>
    </Modal>
  );
};
