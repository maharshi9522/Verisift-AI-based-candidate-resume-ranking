import React from 'react';
import { useAuth } from '../hooks/useAuth';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { User, Mail, Calendar, Key, ShieldCheck } from 'lucide-react';

export const Profile: React.FC = () => {
  const { user } = useAuth();

  const formattedDate = user?.created_at
    ? new Date(user.created_at).toLocaleString('en-US', {
        dateStyle: 'medium',
        timeStyle: 'short',
      })
    : 'Active';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '780px' }}>
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
          Settings
        </h1>
        <p style={{ fontSize: '13.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
          Recruiter identity and workspace credentials.
        </p>
      </div>

      {/* Recruiter Identity Card */}
      <Card
        title="Recruiter Identity"
        subtitle="Verified credentials associated with your hiring workspace."
        headerAction={<Badge variant="success">Active</Badge>}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '14px',
            marginTop: '8px',
          }}
        >
          <div
            style={{
              padding: '14px 16px',
              backgroundColor: 'var(--color-hover)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--color-border)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
              <User size={14} />
              <span style={{ fontSize: '11.5px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Full Name
              </span>
            </div>
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--color-text)' }}>
              {user?.full_name || '—'}
            </div>
          </div>

          <div
            style={{
              padding: '14px 16px',
              backgroundColor: 'var(--color-hover)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--color-border)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
              <Mail size={14} />
              <span style={{ fontSize: '11.5px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Work Email
              </span>
            </div>
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--color-text)' }}>
              {user?.email || '—'}
            </div>
          </div>

          <div
            style={{
              padding: '14px 16px',
              backgroundColor: 'var(--color-hover)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--color-border)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
              <Calendar size={14} />
              <span style={{ fontSize: '11.5px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Account Created
              </span>
            </div>
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-text)' }}>
              {formattedDate}
            </div>
          </div>

          <div
            style={{
              padding: '14px 16px',
              backgroundColor: 'var(--color-hover)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--color-border)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
              <Key size={14} />
              <span style={{ fontSize: '11.5px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Account ID
              </span>
            </div>
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-primary)' }}>
              #{user?.id ?? '1'}
            </div>
          </div>
        </div>
      </Card>

      {/* Security & Access Card */}
      <Card
        title="Security & Storage"
        subtitle="Enterprise data isolation standards for candidate information."
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
            <ShieldCheck size={16} color="#16866A" style={{ marginTop: '2px', flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-text)' }}>
                Encrypted Credentials & Isolated Storage
              </div>
              <div style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', marginTop: '2px', lineHeight: 1.5 }}>
                Passwords are protected using cryptographic salted hashes (bcrypt cost 12). Sessions are secured via authenticated bearer tokens and localized database persistence.
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};
