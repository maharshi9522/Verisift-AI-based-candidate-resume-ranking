import React from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Briefcase, ArrowRight } from 'lucide-react';

interface CreateRoleModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CreateRoleModal: React.FC<CreateRoleModalProps> = ({ isOpen, onClose }) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Create a role"
      description="Define role competencies, experience criteria, and responsibilities."
      footer={
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={onClose} rightIcon={<ArrowRight size={14} />}>
            Understood
          </Button>
        </div>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div
          style={{
            padding: '16px',
            backgroundColor: 'var(--color-hover)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--color-border)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px',
          }}
        >
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-xs)',
              backgroundColor: 'rgba(39, 211, 176, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#12343B',
              flexShrink: 0,
            }}
          >
            <Briefcase size={16} />
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-primary)', marginBottom: '2px' }}>
              Role Management Coming in Next Release
            </div>
            <p style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
              Role definition and resume ingestion pipelines are being connected. In the next release, you will be able to input job descriptions, set required competencies, and upload candidate resumes in bulk.
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ fontSize: '12.5px', fontWeight: 500, color: 'var(--color-text)' }}>
            What you'll be able to do:
          </div>
          <ul style={{ paddingLeft: '18px', fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
            <li>Define roles with mandatory and preferred competency weights</li>
            <li>Upload multiple PDF and DOCX candidate resumes simultaneously</li>
            <li>Evaluate contextual fit beyond surface-level keyword frequency</li>
          </ul>
        </div>
      </div>
    </Modal>
  );
};
