import React from 'react';

export type BadgeVariant = 'turquoise' | 'teal' | 'success' | 'warning' | 'error' | 'muted';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  icon?: React.ReactNode;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'turquoise',
  icon,
  className = '',
}) => {
  const styles: Record<BadgeVariant, React.CSSProperties> = {
    turquoise: {
      backgroundColor: 'rgba(39, 211, 176, 0.14)',
      color: '#12343B',
      border: '1px solid rgba(39, 211, 176, 0.35)',
      fontWeight: 600,
    },
    teal: {
      backgroundColor: 'var(--color-hover)',
      color: '#12343B',
      border: '1px solid var(--color-border)',
      fontWeight: 500,
    },
    success: {
      backgroundColor: 'var(--color-success-bg)',
      color: 'var(--color-success)',
      border: '1px solid rgba(22, 134, 106, 0.2)',
      fontWeight: 500,
    },
    warning: {
      backgroundColor: 'var(--color-warning-bg)',
      color: 'var(--color-warning)',
      border: '1px solid rgba(166, 107, 18, 0.25)',
      fontWeight: 500,
    },
    error: {
      backgroundColor: 'var(--color-error-bg)',
      color: 'var(--color-error)',
      border: '1px solid rgba(195, 68, 68, 0.25)',
      fontWeight: 500,
    },
    muted: {
      backgroundColor: '#E8EEF0',
      color: 'var(--color-text-secondary)',
      border: '1px solid var(--color-border)',
      fontWeight: 500,
    },
  };

  return (
    <span
      className={`badge ${className}`.trim()}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '5px',
        padding: '3px 10px',
        borderRadius: 'var(--radius-full)',
        fontSize: '11.5px',
        lineHeight: 1.2,
        letterSpacing: '0.02em',
        ...styles[variant],
      }}
    >
      {icon && <span style={{ display: 'inline-flex' }}>{icon}</span>}
      <span>{children}</span>
    </span>
  );
};
