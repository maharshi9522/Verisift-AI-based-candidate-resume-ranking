import React from 'react';
import { Plus, Search, RotateCcw } from 'lucide-react';
import { Button } from '../ui/Button';

interface RolesEmptyStateProps {
  onCreateRole?: () => void;
  isFiltered?: boolean;
  onClearFilters?: () => void;
}

export const RolesEmptyState: React.FC<RolesEmptyStateProps> = ({
  onCreateRole,
  isFiltered = false,
  onClearFilters,
}) => {
  if (isFiltered) {
    return (
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
          No matching roles
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
          No roles matched your search or status filter. Try adjusting your query or reset filters to see all roles.
        </p>

        {onClearFilters && (
          <Button
            variant="secondary"
            size="md"
            leftIcon={<RotateCcw size={14} />}
            onClick={onClearFilters}
          >
            Clear search & filters
          </Button>
        )}
      </div>
    );
  }

  return (
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
      {/* Editorial Custom Vector Illustration */}
      <div style={{ width: '160px', height: '110px', marginBottom: '20px' }}>
        <svg width="100%" height="100%" viewBox="0 0 160 110" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Background subtle grid base */}
          <rect x="25" y="15" width="110" height="80" rx="8" fill="#F4F8FA" stroke="#DCE6E8" strokeWidth="1" />
          
          {/* Subtle document shape */}
          <rect x="42" y="26" width="76" height="58" rx="6" fill="#FFFFFF" stroke="#DCE6E8" strokeWidth="1" />
          
          {/* Document header lines */}
          <rect x="52" y="36" width="28" height="4" rx="2" fill="#12343B" />
          <rect x="52" y="44" width="46" height="3" rx="1.5" fill="#DCE6E8" />
          <rect x="52" y="51" width="36" height="3" rx="1.5" fill="#DCE6E8" />
          
          {/* Connected requirement node line */}
          <path d="M96 65L120 48" stroke="#27D3B0" strokeWidth="1.5" strokeDasharray="3 3" />
          <circle cx="96" cy="65" r="3.5" fill="#12343B" stroke="#27D3B0" strokeWidth="1.5" />
          <circle cx="120" cy="48" r="3.5" fill="#27D3B0" />

          {/* Accent badge stamp */}
          <circle cx="106" cy="72" r="11" fill="#12343B" />
          <path d="M102 72L105 75L110 69" stroke="#27D3B0" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
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
        No roles yet
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
        Create a role to start organizing your candidate pipeline.
      </p>

      {onCreateRole && (
        <Button
          variant="primary"
          size="md"
          leftIcon={<Plus size={15} />}
          onClick={onCreateRole}
        >
          Create your first role
        </Button>
      )}
    </div>
  );
};
