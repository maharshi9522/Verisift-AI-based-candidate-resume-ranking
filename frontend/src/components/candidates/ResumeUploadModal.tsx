import React, { useState, useRef, useCallback } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { candidateService } from '../../services/candidates';
import type { BatchUploadResponse } from '../../types/candidate';
import {
  UploadCloud,
  FileText,
  X,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Plus,
} from 'lucide-react';

interface ResumeUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  jobId: number;
  jobTitle: string;
  onUploadSuccess: (result: BatchUploadResponse) => void;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export const ResumeUploadModal: React.FC<ResumeUploadModalProps> = ({
  isOpen,
  onClose,
  jobId,
  jobTitle,
  onUploadSuccess,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<BatchUploadResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetState = useCallback(() => {
    setSelectedFiles([]);
    setIsDragging(false);
    setIsUploading(false);
    setUploadResult(null);
    setErrorMessage(null);
  }, []);

  const handleClose = () => {
    if (isUploading) return;
    resetState();
    onClose();
  };

  const handleAddFiles = (newFiles: FileList | File[]) => {
    setErrorMessage(null);
    const validFiles: File[] = [];
    const errors: string[] = [];

    Array.from(newFiles).forEach((file) => {
      const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      if (!['.pdf', '.docx'].includes(ext)) {
        errors.push(`"${file.name}" is not a PDF or DOCX file.`);
        return;
      }
      if (file.size > MAX_FILE_SIZE) {
        errors.push(`"${file.name}" exceeds the 10 MB file size limit.`);
        return;
      }
      // Check if file is already in staging list
      if (!selectedFiles.some((f) => f.name === file.name && f.size === file.size)) {
        validFiles.push(file);
      }
    });

    if (errors.length > 0) {
      setErrorMessage(errors[0]);
    }

    if (validFiles.length > 0) {
      setSelectedFiles((prev) => [...prev, ...validFiles]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleAddFiles(e.dataTransfer.files);
    }
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0 || isUploading) return;
    setIsUploading(true);
    setErrorMessage(null);

    try {
      const result = await candidateService.uploadJobCandidates(jobId, selectedFiles);
      setUploadResult(result);
      onUploadSuccess(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed. Please check network and try again.';
      setErrorMessage(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Upload Resumes"
      description={`Add candidate profiles to ${jobTitle}`}
      maxWidth="640px"
      footer={
        uploadResult ? (
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', width: '100%' }}>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                resetState();
              }}
              leftIcon={<Plus size={14} />}
            >
              Upload more
            </Button>
            <Button variant="primary" size="sm" onClick={handleClose}>
              Done
            </Button>
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
            <span style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)' }}>
              {selectedFiles.length} file{selectedFiles.length === 1 ? '' : 's'} selected
            </span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <Button variant="ghost" size="sm" onClick={handleClose} disabled={isUploading}>
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleUpload}
                disabled={selectedFiles.length === 0 || isUploading}
                isLoading={isUploading}
                leftIcon={<UploadCloud size={15} />}
              >
                Upload {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''}
              </Button>
            </div>
          </div>
        )
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Error banner */}
        {errorMessage && (
          <div
            style={{
              padding: '10px 14px',
              backgroundColor: 'var(--color-error-bg)',
              border: '1px solid rgba(195, 68, 68, 0.25)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--color-error)',
              fontSize: '12.5px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
            role="alert"
          >
            <AlertCircle size={15} style={{ flexShrink: 0 }} />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Upload Summary View (shown after upload finishes) */}
        {uploadResult ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div
              style={{
                padding: '14px 18px',
                backgroundColor: 'var(--color-hover)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border)',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}
            >
              <CheckCircle2 size={22} color="var(--color-primary)" />
              <div>
                <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--color-primary)' }}>
                  Processing Complete
                </div>
                <div style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', marginTop: '2px' }}>
                  {uploadResult.total_uploaded} uploaded &bull; {uploadResult.total_duplicates} duplicate{uploadResult.total_duplicates === 1 ? '' : 's'} &bull; {uploadResult.total_errors} error{uploadResult.total_errors === 1 ? '' : 's'}
                </div>
              </div>
            </div>

            {/* Uploaded items */}
            {uploadResult.uploaded.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-text)' }}>
                  Successfully Ingested ({uploadResult.uploaded.length})
                </span>
                <div style={{ maxHeight: '150px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {uploadResult.uploaded.map((item) => (
                    <div
                      key={item.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '8px 12px',
                        backgroundColor: 'var(--color-surface)',
                        border: '1px solid var(--color-border)',
                        borderRadius: 'var(--radius-xs)',
                        fontSize: '12.5px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                        <FileText size={14} color="var(--color-primary)" />
                        <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                          {item.original_filename}
                        </span>
                      </div>
                      <Badge variant={item.processing_status === 'completed' ? 'turquoise' : 'error'}>
                        {item.processing_status === 'completed' ? 'Extracted' : 'Failed text'}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Duplicate notices */}
            {uploadResult.duplicates.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-warning)' }}>
                  Duplicates Skipped ({uploadResult.duplicates.length})
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {uploadResult.duplicates.map((dup, i) => (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '8px 12px',
                        backgroundColor: 'var(--color-warning-bg)',
                        border: '1px solid rgba(166, 107, 18, 0.25)',
                        borderRadius: 'var(--radius-xs)',
                        fontSize: '12px',
                        color: 'var(--color-warning)',
                      }}
                    >
                      <AlertTriangle size={14} style={{ flexShrink: 0 }} />
                      <span>{dup.filename}: {dup.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error notices */}
            {uploadResult.errors.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-error)' }}>
                  Rejected Files ({uploadResult.errors.length})
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {uploadResult.errors.map((err, i) => (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '8px 12px',
                        backgroundColor: 'var(--color-error-bg)',
                        border: '1px solid rgba(195, 68, 68, 0.25)',
                        borderRadius: 'var(--radius-xs)',
                        fontSize: '12px',
                        color: 'var(--color-error)',
                      }}
                    >
                      <AlertCircle size={14} style={{ flexShrink: 0 }} />
                      <span>{err.filename}: {err.reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Staging & Dropzone View */
          <>
            {/* Drag and Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: isDragging ? '2px dashed var(--color-accent)' : '2px dashed var(--color-border)',
                backgroundColor: isDragging ? 'rgba(39, 211, 176, 0.08)' : 'var(--color-background)',
                borderRadius: 'var(--radius-md)',
                padding: '36px 20px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all var(--transition-fast)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              <div
                style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: 'var(--radius-full)',
                  backgroundColor: 'var(--color-hover)',
                  color: 'var(--color-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '4px',
                }}
              >
                <UploadCloud size={22} />
              </div>

              <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--color-text)' }}>
                Drag & drop candidate resumes here
              </div>

              <p style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
                or <span style={{ color: 'var(--color-primary)', textDecoration: 'underline' }}>browse files</span> from your computer
              </p>

              <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 500,
                    color: 'var(--color-text-muted)',
                    backgroundColor: 'var(--color-surface)',
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-xs)',
                    border: '1px solid var(--color-border)',
                  }}
                >
                  PDF & DOCX only
                </span>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 500,
                    color: 'var(--color-text-muted)',
                    backgroundColor: 'var(--color-surface)',
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-xs)',
                    border: '1px solid var(--color-border)',
                  }}
                >
                  Max 10 MB per file
                </span>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleAddFiles(e.target.files);
                  }
                }}
              />
            </div>

            {/* Selected files staging list */}
            {selectedFiles.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--color-text)' }}>
                    Files to upload ({selectedFiles.length})
                  </span>
                  <button
                    onClick={() => setSelectedFiles([])}
                    style={{
                      background: 'none',
                      border: 'none',
                      fontSize: '11.5px',
                      color: 'var(--color-text-secondary)',
                      cursor: 'pointer',
                    }}
                  >
                    Clear all
                  </button>
                </div>

                <div
                  style={{
                    maxHeight: '180px',
                    overflowY: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '4px',
                    backgroundColor: 'var(--color-surface)',
                  }}
                >
                  {selectedFiles.map((file, idx) => (
                    <div
                      key={`${file.name}-${idx}`}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '6px 10px',
                        borderRadius: 'var(--radius-xs)',
                        backgroundColor: 'var(--color-background)',
                        fontSize: '12.5px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                        <FileText size={14} color="var(--color-primary)" />
                        <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                          {file.name}
                        </span>
                        <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
                          ({formatBytes(file.size)})
                        </span>
                      </div>
                      <button
                        onClick={() => handleRemoveFile(idx)}
                        disabled={isUploading}
                        aria-label={`Remove ${file.name}`}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--color-text-secondary)',
                          cursor: 'pointer',
                          padding: '2px',
                          display: 'flex',
                          alignItems: 'center',
                        }}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Modal>
  );
};
