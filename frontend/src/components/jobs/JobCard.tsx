import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Job } from '../../types/job';
import { Badge } from '../ui/Badge';
import {
  Briefcase,
  MapPin,
  Laptop,
  MoreVertical,
  Edit2,
  Trash2,
  Archive,
  CheckCircle,
  Clock,
  Users,
} from 'lucide-react';

interface JobCardProps {
  job: Job;
  onClick: (job: Job) => void;
  onEdit: (job: Job) => void;
  onDelete: (job: Job) => void;
  onArchive: (job: Job) => void;
  onToggleStatus: (job: Job) => void;
}

export const JobCard: React.FC<JobCardProps> = ({
  job,
  onClick,
  onEdit,
  onDelete,
  onArchive,
  onToggleStatus,
}) => {
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    if (isMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isMenuOpen]);

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return isoString;
    }
  };

  const renderStatusBadge = () => {
    switch (job.status) {
      case 'active':
        return <Badge variant="turquoise">Active</Badge>;
      case 'draft':
        return <Badge variant="muted">Draft</Badge>;
      case 'archived':
        return <Badge variant="warning">Archived</Badge>;
      default:
        return <Badge variant="teal">{job.status}</Badge>;
    }
  };

  const visibleSkills = job.required_skills?.slice(0, 3) || [];
  const remainingSkillsCount = (job.required_skills?.length || 0) - visibleSkills.length;

  return (
    <div
      onClick={() => onClick(job)}
      style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-sm)',
        padding: '16px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        cursor: 'pointer',
        transition: 'border-color var(--transition-fast), box-shadow var(--transition-fast)',
        position: 'relative',
        boxShadow: 'var(--shadow-card)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#B7C8CB';
        e.currentTarget.style.boxShadow = 'var(--shadow-md)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--color-border)';
        e.currentTarget.style.boxShadow = 'var(--shadow-card)';
      }}
    >
      {/* Top Header Row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <h3
              style={{
                fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                fontSize: '16px',
                fontWeight: 600,
                color: 'var(--color-text)',
                margin: 0,
              }}
            >
              {job.title}
            </h3>
            {renderStatusBadge()}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', fontSize: '12.5px', color: 'var(--color-text-secondary)' }}>
            {job.department && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Briefcase size={13} color="#61777D" />
                {job.department}
              </span>
            )}
            {job.location && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <MapPin size={13} color="#61777D" />
                {job.location}
              </span>
            )}
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Laptop size={13} color="#61777D" />
              {job.workplace_type} &bull; {job.employment_type}
            </span>
            {job.minimum_experience !== null && job.minimum_experience !== undefined && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Clock size={13} color="#61777D" />
                {job.minimum_experience}+ yrs exp
              </span>
            )}
          </div>
        </div>

        {/* Action Menu (stops click propagation) */}
        <div
          ref={menuRef}
          style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '6px' }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => navigate(`/candidates?jobId=${job.id}`)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '4px 10px',
              fontSize: '12px',
              fontWeight: 500,
              color: 'var(--color-primary)',
              backgroundColor: 'var(--color-hover)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-xs)',
              cursor: 'pointer',
              transition: 'all var(--transition-fast)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(39, 211, 176, 0.15)';
              e.currentTarget.style.borderColor = 'rgba(39, 211, 176, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--color-hover)';
              e.currentTarget.style.borderColor = 'var(--color-border)';
            }}
          >
            <Users size={13} />
            <span>Candidates</span>
          </button>

          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            aria-label="Job actions"
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--color-text-secondary)',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: 'var(--radius-xs)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background-color var(--transition-fast)',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-hover)')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            <MoreVertical size={16} />
          </button>

          {isMenuOpen && (
            <div
              style={{
                position: 'absolute',
                right: 0,
                top: '100%',
                marginTop: '4px',
                width: '180px',
                backgroundColor: 'var(--color-surface)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border)',
                boxShadow: 'var(--shadow-modal)',
                zIndex: 20,
                padding: '4px',
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
              }}
              className="animate-fade-in"
            >
              <button
                onClick={() => {
                  setIsMenuOpen(false);
                  navigate(`/candidates?jobId=${job.id}`);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  width: '100%',
                  padding: '7px 10px',
                  fontSize: '12.5px',
                  color: 'var(--color-text)',
                  background: 'none',
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-hover)')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <Users size={13} color="var(--color-primary)" />
                <span>View candidates</span>
              </button>

              <button
                onClick={() => {
                  setIsMenuOpen(false);
                  onEdit(job);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  width: '100%',
                  padding: '7px 10px',
                  fontSize: '12.5px',
                  color: 'var(--color-text)',
                  background: 'none',
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-hover)')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <Edit2 size={13} />
                <span>Edit role</span>
              </button>

              <button
                onClick={() => {
                  setIsMenuOpen(false);
                  onToggleStatus(job);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  width: '100%',
                  padding: '7px 10px',
                  fontSize: '12.5px',
                  color: 'var(--color-text)',
                  background: 'none',
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-hover)')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <CheckCircle size={13} />
                <span>{job.status === 'active' ? 'Mark as Draft' : 'Publish Role'}</span>
              </button>

              {job.status !== 'archived' && (
                <button
                  onClick={() => {
                    setIsMenuOpen(false);
                    onArchive(job);
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    width: '100%',
                    padding: '7px 10px',
                    fontSize: '12.5px',
                    color: 'var(--color-text)',
                    background: 'none',
                    border: 'none',
                    borderRadius: 'var(--radius-xs)',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-hover)')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                  <Archive size={13} />
                  <span>Archive</span>
                </button>
              )}

              <div style={{ height: '1px', backgroundColor: 'var(--color-border)', margin: '2px 0' }} />

              <button
                onClick={() => {
                  setIsMenuOpen(false);
                  onDelete(job);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  width: '100%',
                  padding: '7px 10px',
                  fontSize: '12.5px',
                  color: 'var(--color-error)',
                  background: 'none',
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--color-error-bg)')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <Trash2 size={13} />
                <span>Delete role</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Description Snippet */}
      <p
        style={{
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          lineHeight: 1.5,
          margin: 0,
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {job.description}
      </p>

      {/* Skills & Footer Meta */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
          paddingTop: '8px',
          borderTop: '1px solid rgba(220, 230, 232, 0.5)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          {visibleSkills.map((skill) => (
            <span
              key={skill}
              style={{
                fontSize: '11px',
                fontWeight: 500,
                color: 'var(--color-primary)',
                backgroundColor: 'rgba(39, 211, 176, 0.08)',
                padding: '2px 7px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid rgba(39, 211, 176, 0.25)',
              }}
            >
              {skill}
            </span>
          ))}
          {remainingSkillsCount > 0 && (
            <span
              style={{
                fontSize: '11px',
                color: 'var(--color-text-muted)',
                backgroundColor: 'var(--color-hover)',
                padding: '2px 6px',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--color-border)',
              }}
            >
              +{remainingSkillsCount} more
            </span>
          )}
        </div>

        <span style={{ fontSize: '11.5px', color: 'var(--color-text-muted)' }}>
          {formatDate(job.created_at)}
        </span>
      </div>
    </div>
  );
};
