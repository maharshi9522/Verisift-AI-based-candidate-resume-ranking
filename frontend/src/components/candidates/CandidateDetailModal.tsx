import React, { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import type { Candidate, CandidateExplanationData, SkillEvidenceItem, ExperienceHighlightItem } from '../../types/candidate';
import {
  FileText,
  Copy,
  Check,
  RotateCcw,
  Trash2,
  AlertTriangle,
  Calendar,
  Briefcase,
  Layers,
  Award,
  CheckCircle2,
  HelpCircle,
  XCircle,
  ShieldCheck,
  TrendingUp,
  Quote
} from 'lucide-react';

interface CandidateDetailModalProps {
  candidate: Candidate | null;
  isOpen: boolean;
  onClose: () => void;
  onRetry: (candidate: Candidate) => Promise<void>;
  onDelete: (candidate: Candidate) => void;
  isRetrying?: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export const CandidateDetailModal: React.FC<CandidateDetailModalProps> = ({
  candidate,
  isOpen,
  onClose,
  onRetry,
  onDelete,
  isRetrying = false,
}) => {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'evidence' | 'raw_text'>('evidence');

  if (!candidate) return null;

  const explanationData: CandidateExplanationData | null = (() => {
    if (!candidate.explanation) return null;
    try {
      return JSON.parse(candidate.explanation);
    } catch {
      return null;
    }
  })();

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  const handleCopyText = async () => {
    if (!candidate.extracted_text) return;
    try {
      await navigator.clipboard.writeText(candidate.extracted_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const textLength = candidate.extracted_text ? candidate.extracted_text.length : 0;
  const wordCount = candidate.extracted_text
    ? candidate.extracted_text.trim().split(/\s+/).filter(Boolean).length
    : 0;

  const renderStatusBadge = () => {
    switch (candidate.processing_status) {
      case 'completed':
        return <Badge variant="turquoise">Extracted</Badge>;
      case 'processing':
        return <Badge variant="teal">Processing</Badge>;
      case 'failed':
        return <Badge variant="error">Extraction Failed</Badge>;
      default:
        return <Badge variant="muted">{candidate.processing_status}</Badge>;
    }
  };

  const renderSkillStatusBadge = (status: string) => {
    switch (status) {
      case 'Matched':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: 700,
            backgroundColor: '#ECFDF5',
            color: '#059669',
            border: '1px solid #A7F3D0'
          }}>
            <CheckCircle2 size={12} /> Demonstrated Match
          </span>
        );
      case 'Partial':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: 700,
            backgroundColor: '#FEFCE8',
            color: '#CA8A04',
            border: '1px solid #FEF08A'
          }}>
            <HelpCircle size={12} /> Partial Evidence
          </span>
        );
      case 'Not evidenced':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: 700,
            backgroundColor: '#FEF2F2',
            color: '#DC2626',
            border: '1px solid #FECACA'
          }}>
            <XCircle size={12} /> Not Evidenced
          </span>
        );
      case 'Parser limitation':
        return (
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: 600,
            backgroundColor: '#F1F5F9',
            color: '#64748B',
            border: '1px solid #CBD5E1'
          }}>
            <AlertTriangle size={12} /> Parser Limitation
          </span>
        );
      default:
        return <Badge variant="muted">{status}</Badge>;
    }
  };

  const displayName = candidate.full_name || candidate.original_filename.replace(/\.[^/.]+$/, '').replace(/_/g, ' ');

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={displayName}
      description={candidate.job_title ? `Candidate Assessment for ${candidate.job_title}` : undefined}
      maxWidth="860px"
      footer={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDelete(candidate)}
            leftIcon={<Trash2 size={14} color="var(--color-error)" />}
            style={{ color: 'var(--color-error)' }}
          >
            Remove candidate
          </Button>

          <div style={{ display: 'flex', gap: '8px' }}>
            {candidate.processing_status === 'failed' && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onRetry(candidate)}
                isLoading={isRetrying}
                leftIcon={<RotateCcw size={14} />}
              >
                Retry text extraction
              </Button>
            )}

            {candidate.extracted_text && (
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyText}
                leftIcon={copied ? <Check size={14} color="var(--color-success)" /> : <Copy size={14} />}
              >
                {copied ? 'Copied to clipboard' : 'Copy resume text'}
              </Button>
            )}

            <Button variant="primary" size="sm" onClick={onClose}>
              Done
            </Button>
          </div>
        </div>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '10px', borderBottom: '1px solid var(--color-border)', paddingBottom: '10px' }}>
          <button
            onClick={() => setActiveTab('evidence')}
            style={{
              background: 'none',
              border: 'none',
              padding: '6px 14px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              borderRadius: 'var(--radius-xs)',
              backgroundColor: activeTab === 'evidence' ? 'var(--color-hover)' : 'transparent',
              color: activeTab === 'evidence' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              borderBottom: activeTab === 'evidence' ? '2px solid var(--color-primary)' : '2px solid transparent'
            }}
          >
            Explainable Fit & Evidence
          </button>
          <button
            onClick={() => setActiveTab('raw_text')}
            style={{
              background: 'none',
              border: 'none',
              padding: '6px 14px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              borderRadius: 'var(--radius-xs)',
              backgroundColor: activeTab === 'raw_text' ? 'var(--color-hover)' : 'transparent',
              color: activeTab === 'raw_text' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              borderBottom: activeTab === 'raw_text' ? '2px solid var(--color-primary)' : '2px solid transparent'
            }}
          >
            Raw Resume Text
          </button>
        </div>

        {/* TAB 1: EXPLAINABILITY & EVIDENCE */}
        {activeTab === 'evidence' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            {/* Overall Fit Banner */}
            <div style={{
              backgroundColor: '#F8FAFC',
              border: '1px solid #E2E8F0',
              borderRadius: 'var(--radius-sm)',
              padding: '16px 20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {candidate.ranking_position && (
                    <span style={{
                      backgroundColor: candidate.ranking_position === 1 ? '#0F172A' : '#1E293B',
                      color: candidate.ranking_position === 1 ? '#FDE047' : '#38BDF8',
                      fontWeight: 800,
                      fontSize: '13px',
                      padding: '4px 10px',
                      borderRadius: '8px',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      <Award size={14} /> Rank #{candidate.ranking_position}
                    </span>
                  )}
                  {candidate.relevance_score !== null && candidate.relevance_score !== undefined && (
                    <span style={{
                      backgroundColor: candidate.fit_category === 'Good Fit' ? '#ECFDF5' : candidate.fit_category === 'Potential Fit' ? '#FEFCE8' : '#F1F5F9',
                      color: candidate.fit_category === 'Good Fit' ? '#059669' : candidate.fit_category === 'Potential Fit' ? '#CA8A04' : '#475569',
                      border: `1px solid ${candidate.fit_category === 'Good Fit' ? '#A7F3D0' : candidate.fit_category === 'Potential Fit' ? '#FEF08A' : '#CBD5E1'}`,
                      fontWeight: 700,
                      fontSize: '13px',
                      padding: '4px 10px',
                      borderRadius: '8px'
                    }}>
                      Relevance Score: {candidate.relevance_score.toFixed(1)} / 100 ({candidate.fit_category})
                    </span>
                  )}
                </div>

                <div style={{ fontSize: '11px', color: '#64748B', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <ShieldCheck size={14} color="#059669" />
                  <span>{candidate.model_version || 'Verisift SBERT v2.0'}</span>
                </div>
              </div>

              {explanationData && (
                <>
                  <p style={{ margin: 0, fontSize: '13.5px', lineHeight: 1.55, color: '#1E293B' }}>
                    {explanationData.overall_summary}
                  </p>

                  {explanationData.peer_comparison && (
                    <div style={{
                      backgroundColor: '#EFF6FF',
                      border: '1px solid #BFDBFE',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      fontSize: '12.5px',
                      color: '#1E40AF',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <TrendingUp size={15} style={{ flexShrink: 0 }} />
                      <span><b>Comparative Ranking Insight:</b> {explanationData.peer_comparison}</span>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Score Breakdown Cards */}
            {explanationData?.score_breakdown && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Transparent Score Breakdown
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
                  <div style={{ padding: '12px', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}>
                    <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>Semantic SBERT (50%)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#0F172A', marginTop: '2px' }}>
                      {explanationData.score_breakdown.semantic_similarity_score.toFixed(1)}
                    </div>
                    <div style={{ fontSize: '11px', color: '#94A3B8' }}>Transformer text relevance</div>
                  </div>

                  <div style={{ padding: '12px', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}>
                    <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>Required Skills (25%)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#059669', marginTop: '2px' }}>
                      {explanationData.score_breakdown.required_skill_coverage.toFixed(0)}%
                    </div>
                    <div style={{ fontSize: '11px', color: '#94A3B8' }}>Mandatory role coverage</div>
                  </div>

                  <div style={{ padding: '12px', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}>
                    <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>Preferred Skills (10%)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#2563EB', marginTop: '2px' }}>
                      {explanationData.score_breakdown.preferred_skill_coverage.toFixed(0)}%
                    </div>
                    <div style={{ fontSize: '11px', color: '#94A3B8' }}>Bonus competencies</div>
                  </div>

                  <div style={{ padding: '12px', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '8px' }}>
                    <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>Experience Evidence (15%)</div>
                    <div style={{ fontSize: '18px', fontWeight: 800, color: '#7C3AED', marginTop: '2px' }}>
                      {explanationData.score_breakdown.experience_project_score.toFixed(0)}
                    </div>
                    <div style={{ fontSize: '11px', color: '#94A3B8' }}>Internships & live projects</div>
                  </div>
                </div>
                <span style={{ fontSize: '11px', color: '#64748B', fontStyle: 'italic' }}>
                  {explanationData.score_breakdown.formula_description}
                </span>
              </div>
            )}

            {/* Required Skills Analysis */}
            {explanationData?.required_skills && explanationData.required_skills.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Required Competency Analysis & Resume Evidence
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {explanationData.required_skills.map((skill: SkillEvidenceItem, idx: number) => (
                    <div
                      key={idx}
                      style={{
                        padding: '12px 14px',
                        backgroundColor: '#FFFFFF',
                        border: '1px solid #E2E8F0',
                        borderRadius: '6px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '6px'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontWeight: 700, fontSize: '13.5px', color: '#0F172A' }}>{skill.skill_name}</span>
                          {skill.section_found && (
                            <span style={{ fontSize: '10.5px', backgroundColor: '#F1F5F9', color: '#64748B', padding: '1px 6px', borderRadius: '4px' }}>
                              Found in: {skill.section_found}
                            </span>
                          )}
                        </div>
                        {renderSkillStatusBadge(skill.status)}
                      </div>

                      {skill.evidence_text ? (
                        <div style={{
                          backgroundColor: '#F8FAFC',
                          borderLeft: '3px solid #38BDF8',
                          padding: '6px 10px',
                          fontSize: '12px',
                          color: '#334155',
                          fontStyle: 'italic',
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '6px'
                        }}>
                          <Quote size={13} style={{ flexShrink: 0, marginTop: '2px', color: '#94A3B8' }} />
                          <span>"{skill.evidence_text}"</span>
                        </div>
                      ) : (
                        <div style={{ fontSize: '12px', color: '#94A3B8' }}>
                          {skill.reason || 'No documented mention in parsed resume text.'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Preferred Skills Analysis */}
            {explanationData?.preferred_skills && explanationData.preferred_skills.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Preferred / Optional Skills
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {explanationData.preferred_skills.map((skill: SkillEvidenceItem, idx: number) => (
                    <div
                      key={idx}
                      style={{
                        padding: '6px 10px',
                        backgroundColor: '#FFFFFF',
                        border: '1px solid #E2E8F0',
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '12px'
                      }}
                    >
                      <span style={{ fontWeight: 600, color: '#1E293B' }}>{skill.skill_name}:</span>
                      {renderSkillStatusBadge(skill.status)}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Practical Experience & Internships */}
            {explanationData?.experience_highlights && explanationData.experience_highlights.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Demonstrated Internships & Practical Projects
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {explanationData.experience_highlights.map((exp: ExperienceHighlightItem, idx: number) => (
                    <div
                      key={idx}
                      style={{
                        padding: '12px 14px',
                        backgroundColor: '#FFFFFF',
                        border: '1px solid #E2E8F0',
                        borderRadius: '6px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '6px'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
                        <span style={{ fontWeight: 700, fontSize: '13px', color: '#0F172A' }}>
                          {exp.role_or_project}
                        </span>
                        {exp.is_internship && (
                          <span style={{
                            backgroundColor: '#EFF6FF',
                            color: '#2563EB',
                            fontSize: '11px',
                            fontWeight: 700,
                            padding: '2px 8px',
                            borderRadius: '10px',
                            border: '1px solid #BFDBFE'
                          }}>
                            Internship Experience
                          </span>
                        )}
                      </div>

                      <p style={{ margin: 0, fontSize: '12.5px', color: '#475569', lineHeight: 1.5 }}>
                        {exp.relevance_summary}
                      </p>

                      {exp.technologies_evidenced && exp.technologies_evidenced.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '2px' }}>
                          {exp.technologies_evidenced.map((tech: string, tIdx: number) => (
                            <span key={tIdx} style={{ fontSize: '10.5px', padding: '1px 6px', backgroundColor: '#F1F5F9', color: '#475569', borderRadius: '4px' }}>
                              {tech}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: RAW TEXT */}
        {activeTab === 'raw_text' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* Metadata Strip */}
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'center',
                gap: '12px',
                paddingBottom: '12px',
                borderBottom: '1px solid var(--color-border)',
                fontSize: '12.5px',
                color: 'var(--color-text-secondary)',
              }}
            >
              {renderStatusBadge()}

              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <FileText size={14} color="#61777D" />
                <span style={{ textTransform: 'uppercase', fontWeight: 600 }}>{candidate.file_type}</span>
                <span>&bull; {formatBytes(candidate.file_size)}</span>
              </div>

              {candidate.job_title && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <Briefcase size={14} color="#61777D" />
                  <span>{candidate.job_title}</span>
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginLeft: 'auto' }}>
                <Calendar size={13} color="#8E9FA4" />
                <span>Uploaded {formatDate(candidate.created_at)}</span>
              </div>
            </div>

            {/* Extracted Resume Text Pane */}
            {candidate.processing_status === 'completed' && candidate.extracted_text && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Layers size={14} color="var(--color-primary)" />
                    <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-text)' }}>
                      Complete Parsed Text
                    </span>
                  </div>
                  <span style={{ fontSize: '11.5px', color: 'var(--color-text-muted)' }}>
                    {wordCount.toLocaleString()} words &bull; {textLength.toLocaleString()} characters
                  </span>
                </div>

                <div
                  style={{
                    backgroundColor: 'var(--color-background)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '16px',
                    maxHeight: '450px',
                    overflowY: 'auto',
                    fontFamily: 'inherit',
                    fontSize: '12.5px',
                    lineHeight: 1.6,
                    color: 'var(--color-text)',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {candidate.extracted_text}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  );
};
