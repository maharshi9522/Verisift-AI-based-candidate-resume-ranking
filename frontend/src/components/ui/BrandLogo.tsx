import React from 'react';

interface BrandLogoProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'light' | 'dark'; // 'light' is for dark backgrounds (white text), 'dark' is for light backgrounds (teal text)
  showTagline?: boolean;
}

export const BrandLogo: React.FC<BrandLogoProps> = ({
  size = 'md',
  variant = 'dark',
  showTagline = false,
}) => {
  const iconSizes = {
    sm: 28,
    md: 36,
    lg: 44,
  };

  const fontSizes = {
    sm: '17px',
    md: '21px',
    lg: '26px',
  };

  const currentIconSize = iconSizes[size];
  const currentFontSize = fontSizes[size];
  const textColor = variant === 'light' ? '#FFFFFF' : '#12343B';
  const subtextColor = variant === 'light' ? '#A5B5B9' : '#61777D';

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', userSelect: 'none' }}>
      {/* Brand Icon Mark */}
      <div
        style={{
          width: `${currentIconSize}px`,
          height: `${currentIconSize}px`,
          borderRadius: '10px',
          backgroundColor: '#12343B',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          border: '1px solid rgba(39, 211, 176, 0.25)',
          boxShadow: '0 2px 6px rgba(18, 52, 59, 0.15)',
        }}
      >
        <svg
          width={Math.round(currentIconSize * 0.65)}
          height={Math.round(currentIconSize * 0.65)}
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Subtle document frame */}
          <rect x="3" y="3" width="18" height="18" rx="4" stroke="#27D3B0" strokeWidth="1.5" strokeOpacity="0.4" />
          {/* Contextual inspection / scanning loop */}
          <circle cx="11" cy="11" r="4.5" stroke="#27D3B0" strokeWidth="2" strokeLinecap="round" />
          <line x1="14.5" y1="14.5" x2="18.5" y2="18.5" stroke="#27D3B0" strokeWidth="2" strokeLinecap="round" />
          {/* Verification check inside */}
          <path d="M9.5 11L10.75 12.25L13 9.75" stroke="#27D3B0" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {/* Brand Wordmark */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <span
          style={{
            fontFamily: "var(--font-heading, 'Sora', sans-serif)",
            fontWeight: 600,
            fontSize: currentFontSize,
            lineHeight: 1.1,
            color: textColor,
            letterSpacing: '-0.02em',
          }}
        >
          Verisift
        </span>
        {showTagline && (
          <span
            style={{
              fontFamily: "var(--font-body, 'Inter', sans-serif)",
              fontSize: '11px',
              fontWeight: 400,
              color: subtextColor,
              letterSpacing: '0.01em',
              marginTop: '2px',
            }}
          >
            Find fit beyond keywords
          </span>
        )}
      </div>
    </div>
  );
};
