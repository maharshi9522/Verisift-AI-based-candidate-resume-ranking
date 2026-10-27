import React from 'react';
import { Menu, User as UserIcon, LogOut } from 'lucide-react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

interface TopBarProps {
  onToggleSidebar: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ onToggleSidebar }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user: authUser, logout } = useAuth();

  const getPageTitle = (pathname: string) => {
    switch (pathname) {
      case '/':
        return 'Overview';
      case '/jobs':
        return 'Roles';
      case '/candidates':
        return 'Candidates';
      case '/shortlist':
        return 'Shortlist';
      case '/settings':
        return 'Settings';
      default:
        return 'Workspace';
    }
  };

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <header
      style={{
        height: '52px',
        backgroundColor: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 30,
      }}
    >
      {/* Left: Mobile Drawer Trigger & Context Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <button
          onClick={onToggleSidebar}
          aria-label="Open navigation menu"
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--color-text)',
            cursor: 'pointer',
            padding: '6px',
            borderRadius: 'var(--radius-xs)',
            display: 'none',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          className="mobile-menu-trigger"
        >
          <Menu size={20} />
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)', fontWeight: 400 }}>
            Workspace
          </span>
          <span style={{ fontSize: '12px', color: 'var(--color-border)' }}>/</span>
          <span style={{ fontSize: '13.5px', color: 'var(--color-text)', fontWeight: 600 }}>
            {getPageTitle(location.pathname)}
          </span>
        </div>
      </div>

      {/* Right: Functional Recruiter Identity & Sign Out */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Link
          to="/settings"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '4px 10px',
            borderRadius: 'var(--radius-xs)',
            border: '1px solid var(--color-border)',
            backgroundColor: 'var(--color-surface)',
            color: 'var(--color-text)',
            fontSize: '12.5px',
            fontWeight: 500,
            textDecoration: 'none',
            transition: 'background-color var(--transition-fast), border-color var(--transition-fast)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--color-hover)';
            e.currentTarget.style.borderColor = '#CBD8DB';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--color-surface)';
            e.currentTarget.style.borderColor = 'var(--color-border)';
          }}
        >
          <UserIcon size={14} color="#12343B" />
          <span>{authUser?.full_name?.split(' ')[0] || 'Recruiter'}</span>
        </Link>

        <button
          onClick={handleLogout}
          title="Sign out"
          aria-label="Sign out"
          style={{
            background: 'none',
            border: '1px solid transparent',
            color: 'var(--color-text-secondary)',
            cursor: 'pointer',
            padding: '6px 8px',
            borderRadius: 'var(--radius-xs)',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '12.5px',
            transition: 'all var(--transition-fast)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--color-hover)';
            e.currentTarget.style.color = 'var(--color-text)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = 'var(--color-text-secondary)';
          }}
        >
          <LogOut size={14} />
          <span className="signout-text">Sign out</span>
        </button>
      </div>

      <style>{`
        @media (max-width: 900px) {
          .mobile-menu-trigger {
            display: flex !important;
          }
        }
        @media (max-width: 500px) {
          .signout-text {
            display: none !important;
          }
        }
      `}</style>
    </header>
  );
};
