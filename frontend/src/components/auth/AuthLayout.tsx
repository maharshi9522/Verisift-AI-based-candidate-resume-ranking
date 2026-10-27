import React from 'react';
import { BrandLogo } from '../ui/BrandLogo';

interface AuthLayoutProps {
  children: React.ReactNode;
}

export const AuthLayout: React.FC<AuthLayoutProps> = ({ children }) => {
  return (
    <div
      style={{
        minHeight: '100vh',
        width: '100%',
        backgroundColor: '#0D272C',
        backgroundImage: `
          radial-gradient(circle at 50% 12%, rgba(39, 211, 176, 0.10) 0%, transparent 55%),
          radial-gradient(circle at 85% 80%, rgba(39, 211, 176, 0.05) 0%, transparent 45%),
          linear-gradient(180deg, #0D272C 0%, #12343B 100%)
        `,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '36px 16px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Ambient Grid Schematic Watermark */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          backgroundImage: `
            linear-gradient(to right, rgba(220, 230, 232, 0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(220, 230, 232, 0.03) 1px, transparent 1px)
          `,
          backgroundSize: '48px 48px',
        }}
      />

      {/* Decorative Candidate Alignment Graphic (Quiet, Ambient Behind Card) */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -52%)',
          width: '720px',
          height: '540px',
          pointerEvents: 'none',
          opacity: 0.12,
        }}
      >
        <svg width="100%" height="100%" viewBox="0 0 720 540" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Subtle connecting vectors */}
          <line x1="80" y1="120" x2="360" y2="270" stroke="#27D3B0" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="640" y1="140" x2="360" y2="270" stroke="#27D3B0" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="120" y1="440" x2="360" y2="270" stroke="#27D3B0" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="600" y1="420" x2="360" y2="270" stroke="#27D3B0" strokeWidth="1" strokeDasharray="4 4" />
          {/* Nodes */}
          <circle cx="80" cy="120" r="4" fill="#27D3B0" />
          <circle cx="640" cy="140" r="4" fill="#27D3B0" />
          <circle cx="120" cy="440" r="4" fill="#27D3B0" />
          <circle cx="600" cy="420" r="4" fill="#27D3B0" />
          <circle cx="360" cy="270" r="6" stroke="#27D3B0" strokeWidth="1.5" />
        </svg>
      </div>

      {/* Main Container */}
      <div
        style={{
          width: '100%',
          maxWidth: '430px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          position: 'relative',
          zIndex: 10,
        }}
        className="animate-slide-up"
      >
        {/* Brand Header */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '28px',
            textAlign: 'center',
          }}
        >
          <BrandLogo size="lg" variant="light" />
          <p
            style={{
              fontFamily: "var(--font-body, 'Inter', sans-serif)",
              fontSize: '13px',
              color: '#93AAB0',
              marginTop: '8px',
              letterSpacing: '0.01em',
            }}
          >
            Find fit beyond keywords
          </p>
        </div>

        {/* Elevated White Form Card */}
        <div
          style={{
            width: '100%',
            backgroundColor: 'var(--color-surface)',
            border: '1px solid rgba(220, 230, 232, 0.9)',
            borderRadius: 'var(--radius-lg)',
            padding: '32px 30px',
            boxShadow: 'var(--shadow-auth)',
          }}
        >
          {children}
        </div>

        {/* Quiet Trust Indicator */}
        <div
          style={{
            marginTop: '24px',
            fontSize: '11.5px',
            color: '#718C92',
            textAlign: 'center',
            letterSpacing: '0.01em',
          }}
        >
          Candidate Intelligence & Resume Verification
        </div>
      </div>
    </div>
  );
};
