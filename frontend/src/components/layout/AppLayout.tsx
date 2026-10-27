import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';

export const AppLayout: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--color-background)' }}>
      {/* Navigation Rail / Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      {/* Main Workspace */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <TopBar onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
        <main
          style={{
            flex: 1,
            padding: '28px 32px',
            maxWidth: '1360px',
            width: '100%',
            margin: '0 auto',
          }}
          className="app-main-content animate-fade-in"
        >
          <Outlet />
        </main>
      </div>

      <style>{`
        @media (max-width: 768px) {
          .app-main-content {
            padding: 20px 16px !important;
          }
        }
      `}</style>
    </div>
  );
};
