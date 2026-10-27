import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Loader2 } from 'lucide-react';
import { BrandLogo } from '../components/ui/BrandLogo';

export const PublicRoute: React.FC<{ children: React.ReactElement }> = ({ children }) => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'var(--color-background)',
          gap: '16px',
        }}
      >
        <BrandLogo size="md" variant="dark" />
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--color-text-secondary)', fontSize: '13px' }}>
          <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
          <span>Loading workspace...</span>
        </div>
      </div>
    );
  }

  if (user) {
    const from = (location.state as any)?.from?.pathname || '/';
    return <Navigate to={from} replace />;
  }

  return children;
};
