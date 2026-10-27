import React from 'react';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  headerAction,
  children,
  className = '',
  style,
  ...props
}) => {
  return (
    <div
      className={`card ${className}`.trim()}
      style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-md)',
        padding: '24px',
        boxShadow: 'var(--shadow-card)',
        ...style,
      }}
      {...props}
    >
      {(title || headerAction) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            marginBottom: '18px',
            gap: '12px',
          }}
        >
          <div>
            {title && (
              <h3
                style={{
                  fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                  fontSize: '17px',
                  fontWeight: 600,
                  color: 'var(--color-text)',
                  marginBottom: subtitle ? '4px' : '0',
                }}
              >
                {title}
              </h3>
            )}
            {subtitle && (
              <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', margin: 0 }}>
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
};
