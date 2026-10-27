import React from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { AlertTriangle } from 'lucide-react';

interface DeleteConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  jobTitle: string;
  isDeleting: boolean;
}

export const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  jobTitle,
  isDeleting,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={isDeleting ? () => {} : onClose}
      title="Delete job role"
      maxWidth="460px"
      footer={
        <div style={{ display: 'flex', gap: '10px' }}>
          <Button
            variant="secondary"
            size="sm"
            onClick={onClose}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={onConfirm}
            isLoading={isDeleting}
            style={{
              backgroundColor: 'var(--color-error)',
              borderColor: 'var(--color-error)',
              color: '#FFFFFF',
            }}
          >
            Delete role
          </Button>
        </div>
      }
    >
      <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--color-error-bg)',
            color: 'var(--color-error)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            border: '1px solid rgba(195, 68, 68, 0.2)',
          }}
        >
          <AlertTriangle size={20} />
        </div>
        <div>
          <p style={{ fontSize: '13.5px', color: 'var(--color-text)', lineHeight: 1.5, margin: 0 }}>
            Are you sure you want to permanently delete{' '}
            <strong style={{ color: 'var(--color-primary)' }}>{jobTitle}</strong>?
          </p>
          <p
            style={{
              fontSize: '12.5px',
              color: 'var(--color-text-secondary)',
              marginTop: '8px',
              lineHeight: 1.4,
              margin: '8px 0 0 0',
            }}
          >
            This action will remove the role configuration and cannot be undone.
          </p>
        </div>
      </div>
    </Modal>
  );
};
