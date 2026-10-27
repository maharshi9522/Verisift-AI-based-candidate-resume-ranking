import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  Compass,
  Briefcase,
  Users,
  BookmarkCheck,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
  X,
} from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  isCollapsed,
  onToggleCollapse,
}) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [hoveredTooltip, setHoveredTooltip] = useState<string | null>(null);

  const navItems = [
    { name: 'Overview', to: '/', icon: <Compass size={18} /> },
    { name: 'Jobs', to: '/jobs', icon: <Briefcase size={18} /> },
    { name: 'Candidates', to: '/candidates', icon: <Users size={18} /> },
    { name: 'Shortlist', to: '/shortlist', icon: <BookmarkCheck size={18} /> },
    { name: 'Settings', to: '/settings', icon: <Settings size={18} /> },
  ];

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const getInitials = (name?: string) => {
    if (!name) return 'R';
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase()
      .substring(0, 2);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(8, 27, 31, 0.45)',
            backdropFilter: 'blur(2px)',
            zIndex: 40,
            display: 'block',
          }}
          className="sidebar-backdrop"
        />
      )}

      <aside
        style={{
          width: isCollapsed ? '68px' : '204px',
          backgroundColor: '#12343B',
          color: '#FFFFFF',
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          borderRight: '1px solid rgba(220, 230, 232, 0.08)',
          transition: 'width var(--transition-smooth), transform var(--transition-normal)',
          flexShrink: 0,
          userSelect: 'none',
        }}
        className={`app-sidebar ${isOpen ? 'open' : ''}`}
      >
        {/* Brand Header */}
        <div
          style={{
            height: '56px',
            padding: isCollapsed ? '0' : '0 18px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: isCollapsed ? 'center' : 'space-between',
            borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Small Brand Icon */}
            <div
              style={{
                width: '30px',
                height: '30px',
                borderRadius: '8px',
                backgroundColor: '#0D272C',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid rgba(39, 211, 176, 0.3)',
                flexShrink: 0,
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <circle cx="11" cy="11" r="5" stroke="#27D3B0" strokeWidth="2" strokeLinecap="round" />
                <line x1="15" y1="15" x2="19" y2="19" stroke="#27D3B0" strokeWidth="2" strokeLinecap="round" />
                <path d="M9 11L10.5 12.5L13.5 9.5" stroke="#27D3B0" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            {!isCollapsed && (
              <span
                style={{
                  fontFamily: "var(--font-heading, 'Sora', sans-serif)",
                  fontSize: '17px',
                  fontWeight: 600,
                  letterSpacing: '-0.02em',
                  color: '#FFFFFF',
                }}
              >
                Verisift
              </span>
            )}
          </div>

          {/* Mobile close button */}
          <button
            onClick={onClose}
            className="mobile-close-btn"
            aria-label="Close menu"
            style={{
              background: 'none',
              border: 'none',
              color: '#A5B5B9',
              cursor: 'pointer',
              padding: '4px',
              display: 'none',
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Minimal Navigation Section */}
        <div style={{ flex: 1, padding: isCollapsed ? '16px 8px' : '16px 10px', overflowY: 'auto' }}>
          <nav style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {navItems.map((item) => (
              <div
                key={item.to}
                style={{ position: 'relative' }}
                onMouseEnter={() => isCollapsed && setHoveredTooltip(item.name)}
                onMouseLeave={() => setHoveredTooltip(null)}
              >
                <NavLink
                  to={item.to}
                  end={item.to === '/'}
                  onClick={() => onClose()}
                  style={({ isActive }) => ({
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: isCollapsed ? 'center' : 'flex-start',
                    gap: '12px',
                    padding: isCollapsed ? '9px 0' : '9px 12px',
                    borderRadius: 'var(--radius-xs)',
                    fontSize: '13px',
                    fontWeight: isActive ? 600 : 500,
                    color: isActive ? '#FFFFFF' : '#A3B9BE',
                    backgroundColor: isActive ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
                    textDecoration: 'none',
                    transition: 'all var(--transition-fast)',
                    position: 'relative',
                  })}
                  className="sidebar-nav-link"
                >
                  {({ isActive }) => (
                    <>
                      {/* Active indicator bar */}
                      {isActive && (
                        <span
                          style={{
                            position: 'absolute',
                            left: 0,
                            top: '6px',
                            bottom: '6px',
                            width: '3px',
                            backgroundColor: '#27D3B0',
                            borderRadius: '0 3px 3px 0',
                          }}
                        />
                      )}
                      <span style={{ color: isActive ? '#27D3B0' : '#8FA7AD', display: 'flex', alignItems: 'center' }}>
                        {item.icon}
                      </span>
                      {!isCollapsed && <span>{item.name}</span>}
                    </>
                  )}
                </NavLink>

                {/* Tooltip when collapsed */}
                {isCollapsed && hoveredTooltip === item.name && (
                  <div
                    style={{
                      position: 'absolute',
                      left: '72px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      backgroundColor: '#0D272C',
                      color: '#FFFFFF',
                      fontSize: '12px',
                      fontWeight: 500,
                      padding: '4px 9px',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid rgba(220, 230, 232, 0.15)',
                      whiteSpace: 'nowrap',
                      zIndex: 100,
                      boxShadow: 'var(--shadow-sm)',
                      pointerEvents: 'none',
                    }}
                    className="animate-fade-in"
                  >
                    {item.name}
                  </div>
                )}
              </div>
            ))}
          </nav>
        </div>

        {/* Rail Collapse Toggle (Desktop Only) */}
        <div
          className="desktop-rail-toggle"
          style={{
            padding: '8px',
            display: 'flex',
            justifyContent: isCollapsed ? 'center' : 'flex-end',
            borderTop: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <button
            onClick={onToggleCollapse}
            aria-label={isCollapsed ? 'Expand navigation' : 'Collapse navigation'}
            title={isCollapsed ? 'Expand navigation rail' : 'Collapse navigation rail'}
            style={{
              background: 'none',
              border: 'none',
              color: '#8FA7AD',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: 'var(--radius-xs)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all var(--transition-fast)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#FFFFFF';
              e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.07)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#8FA7AD';
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        {/* User Footer Profile & Functional Logout */}
        <div
          style={{
            padding: isCollapsed ? '12px 8px' : '12px 14px',
            borderTop: '1px solid rgba(255, 255, 255, 0.06)',
            backgroundColor: 'rgba(0, 0, 0, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: isCollapsed ? 'center' : 'space-between',
            gap: '8px',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '9px',
              overflow: 'hidden',
              cursor: 'pointer',
            }}
            onClick={() => {
              navigate('/settings');
              onClose();
            }}
            title={user ? `${user.full_name} (${user.email})` : 'Recruiter Profile'}
          >
            <div
              style={{
                width: '30px',
                height: '30px',
                borderRadius: '50%',
                backgroundColor: 'rgba(39, 211, 176, 0.16)',
                color: '#27D3B0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 600,
                fontSize: '11.5px',
                flexShrink: 0,
                border: '1px solid rgba(39, 211, 176, 0.25)',
              }}
            >
              {getInitials(user?.full_name)}
            </div>
            {!isCollapsed && (
              <div style={{ overflow: 'hidden' }}>
                <div
                  style={{
                    fontSize: '12.5px',
                    fontWeight: 600,
                    color: '#FFFFFF',
                    whiteSpace: 'nowrap',
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                  }}
                >
                  {user?.full_name || 'Recruiter'}
                </div>
                <div
                  style={{
                    fontSize: '11px',
                    color: '#8FA7AD',
                    whiteSpace: 'nowrap',
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                  }}
                >
                  {user?.email || 'recruiter@verisift.io'}
                </div>
              </div>
            )}
          </div>

          {!isCollapsed && (
            <button
              onClick={handleLogout}
              title="Sign out"
              aria-label="Sign out of Verisift"
              style={{
                background: 'none',
                border: 'none',
                color: '#8FA7AD',
                cursor: 'pointer',
                padding: '6px',
                borderRadius: 'var(--radius-xs)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all var(--transition-fast)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = '#FFFFFF';
                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.08)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = '#8FA7AD';
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <LogOut size={15} />
            </button>
          )}
        </div>
      </aside>

      <style>{`
        .sidebar-nav-link:hover:not(.active) {
          background-color: rgba(255, 255, 255, 0.05);
          color: #FFFFFF !important;
        }
        @media (max-width: 900px) {
          .app-sidebar {
            position: fixed !important;
            top: 0;
            left: 0;
            bottom: 0;
            width: 220px !important;
            transform: translateX(-100%);
          }
          .app-sidebar.open {
            transform: translateX(0);
          }
          .mobile-close-btn {
            display: block !important;
          }
          .desktop-rail-toggle {
            display: none !important;
          }
        }
      `}</style>
    </>
  );
};
