import React from 'react';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import type { Candidate, CandidateExplanationData } from '../../types/candidate';
import {
  FileText,
  Trash2,
  RotateCcw,
  Eye,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Briefcase,
  CheckCircle,
  AlertCircle,
  Award
} from 'lucide-react';

interface CandidateCardProps {
  candidate: Candidate;
  onClick: (candidate: Candidate) => void;
  onDelete: (candidate: Candidate) => void;
  onRetry: (candidate: Candidate) => void;
  isRetrying?: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export const CandidateCard: React.FC<CandidateCardProps> = ({
  candidate,
  onClick,
  onDelete,
  onRetry,
  isRetrying = false,
}) => {
  const explanationData: CandidateExplanationData | null = React.useMemo(() => {
    if (!candidate.explanation) return null;
    try {
      return JSON.parse(candidate.explanation);
    } catch {
      return null;
    }
  }, [candidate.explanation]);

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

  const renderStatus = () => {
    switch (candidate.processing_status) {
      case 'completed':
        return (
          <Badge variant="turquoise" icon={<CheckCircle2 size={12} />}>
            Extracted
          </Badge>
        );
      case 'processing':
        return (
          <Badge variant="teal" icon={<Clock size={12} />}>
            Processing
          </Badge>
        );
      case 'failed':
        return (
          <Badge variant="error" icon={<AlertTriangle size={12} />}>
            Failed
          </Badge>
        );
      default:
        return <Badge variant="muted">{candidate.processing_status}</Badge>;
    }
  };

  const displayName = candidate.full_name || candidate.original_filename.replace(/\.[^/.]+$/, '').replace(/_/g, ' ');

  return (
    <div
      onClick={() => onClick(candidate)}
      style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-sm)',
        padding: '16px 18px',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        gap: '16px',
        cursor: 'pointer',
        boxShadow: 'var(--shadow-card)',
        transition: 'border-color var(--transition-fast), box-shadow var(--transition-fast)',
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
      {/* Left: Icon & File Meta */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px', minWidth: 0, flex: 1 }}>
        <div
          style={{
            width: '40px',
            height: '40px',
            borderRadius: 'var(--radius-xs)',
            backgroundColor: candidate.ranking_position === 1 ? '#FEF3C7' : 'var(--color-hover)',
            color: candidate.ranking_position === 1 ? '#D97706' : 'var(--color-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            border: `1px solid ${candidate.ranking_position === 1 ? '#FDE68A' : 'var(--color-border)'}`,
          }}
        >
          {candidate.ranking_position === 1 ? <Award size={20} /> : <FileText size={20} />}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span
              style={{
                fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                fontSize: '15px',
                fontWeight: 600,
                color: 'var(--color-text)',
                textOverflow: 'ellipsis',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
              }}
            >
              {displayName}
            </span>

            {candidate.ranking_position && (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '11px',
                  fontWeight: 700,
                  backgroundColor: candidate.ranking_position === 1 ? '#0F172A' : '#1E293B',
                  color: candidate.ranking_position === 1 ? '#FDE047' : '#38BDF8',
                  border: '1px solid #334155'
                }}
                title="Ordinal candidate rank for this job"
              >
                Rank #{candidate.ranking_position}
              </span>
            )}

            {candidate.relevance_score !== null && candidate.relevance_score !== undefined && (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '11px',
                  fontWeight: 700,
                  backgroundColor: candidate.fit_category === 'Good Fit' ? '#ECFDF5' : candidate.fit_category === 'Potential Fit' ? '#FEFCE8' : '#F1F5F9',
                  color: candidate.fit_category === 'Good Fit' ? '#059669' : candidate.fit_category === 'Potential Fit' ? '#CA8A04' : '#475569',
                  border: `1px solid ${candidate.fit_category === 'Good Fit' ? '#A7F3D0' : candidate.fit_category === 'Potential Fit' ? '#FEF08A' : '#CBD5E1'}`
                }}
                title="Relevance score (0-100) based on validated semantic & evidence weights"
              >
                {candidate.relevance_score.toFixed(1)}/100 &bull; {candidate.fit_category}
              </span>
            )}

            {renderStatus()}
          </div>

          {/* Explainability Snippets (Strongest Match & Top Missing Requirement) */}
          {explanationData && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', marginTop: '2px' }}>
              {explanationData.strongest_match && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#047857' }}>
                  <CheckCircle size={13} style={{ flexShrink: 0 }} />
                  <span style={{ fontWeight: 500 }}>{explanationData.strongest_match}</span>
                </div>
              )}
              {explanationData.top_missing_requirement && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#B45309' }}>
                  <AlertCircle size={13} style={{ flexShrink: 0 }} />
                  <span style={{ fontWeight: 500 }}>{explanationData.top_missing_requirement}</span>
                </div>
              )}
            </div>
          )}

          {/* Metadata Row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px', color: 'var(--color-text-secondary)', flexWrap: 'wrap', marginTop: '2px' }}>
            <span style={{ textTransform: 'uppercase', fontWeight: 600, color: 'var(--color-primary)' }}>
              {candidate.file_type}
            </span>
            <span>&bull;</span>
            <span>{candidate.original_filename}</span>
            <span>&bull;</span>
            <span>{formatBytes(candidate.file_size)}</span>

            {candidate.job_title && (
              <>
                <span>&bull;</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Briefcase size={12} color="#61777D" />
                  {candidate.job_title}
                </span>
              </>
            )}

            <span>&bull;</span>
            <span>{formatDate(candidate.created_at)}</span>
          </div>

          {candidate.processing_status === 'failed' && candidate.error_message && (
            <span
              style={{
                fontSize: '11.5px',
                color: 'var(--color-error)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                maxWidth: '450px',
              }}
            >
              {candidate.error_message}
            </span>
          )}
        </div>
      </div>

      {/* Right: Quick Actions */}
      <div
        style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        {candidate.processing_status === 'failed' && (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onRetry(candidate)}
            isLoading={isRetrying}
            leftIcon={<RotateCcw size={13} />}
          >
            Retry
          </Button>
        )}

        <Button
          variant="secondary"
          size="sm"
          onClick={() => onClick(candidate)}
          leftIcon={<Eye size={13} />}
        >
          {candidate.explanation ? 'Review Evidence' : 'Inspect'}
        </Button>

        <button
          onClick={() => onDelete(candidate)}
          aria-label={`Delete ${candidate.original_filename}`}
          title="Remove candidate"
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
            transition: 'all var(--transition-fast)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--color-error)';
            e.currentTarget.style.backgroundColor = 'var(--color-error-bg)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--color-text-secondary)';
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          <Trash2 size={15} />
        </button>
      </div>
    </div>
  );
};
