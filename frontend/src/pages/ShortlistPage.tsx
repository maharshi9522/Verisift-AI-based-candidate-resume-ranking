import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { BookmarkCheck, ArrowRight } from 'lucide-react';

export const ShortlistPage: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
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
            Shortlist
          </h1>
          <p style={{ fontSize: '13.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
            Top candidates flagged for interview progression and hiring manager review.
          </p>
        </div>
      </div>

      {/* Empty State */}
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
          <BookmarkCheck size={22} />
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
          No shortlisted candidates yet
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
          When reviewing applicants for a role, flag strong candidates to compile your hiring shortlist and generate interview briefing summaries.
        </p>

        <Link to="/candidates">
          <Button variant="secondary" size="md" rightIcon={<ArrowRight size={14} />}>
            Go to Candidates
          </Button>
        </Link>
      </div>
    </div>
  );
};
