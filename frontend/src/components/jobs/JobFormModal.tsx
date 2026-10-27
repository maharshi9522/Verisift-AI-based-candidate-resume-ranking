import React, { useState, useEffect } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import type { Job, JobCreateInput, JobUpdateInput, EmploymentType, WorkplaceType, JobStatus } from '../../types/job';
import { jobService } from '../../services/jobs';
import { ApiError } from '../../services/api';
import { Plus, X, AlertCircle } from 'lucide-react';

interface JobFormModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (job: Job, isEdit: boolean) => void;
  jobToEdit?: Job | null;
}

export const JobFormModal: React.FC<JobFormModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  jobToEdit,
}) => {
  const isEditing = !!jobToEdit;

  // Form State
  const [title, setTitle] = useState('');
  const [department, setDepartment] = useState('');
  const [location, setLocation] = useState('');
  const [employmentType, setEmploymentType] = useState<EmploymentType>('Full-time');
  const [workplaceType, setWorkplaceType] = useState<WorkplaceType>('On-site');
  const [description, setDescription] = useState('');
  const [minimumExperience, setMinimumExperience] = useState<string>('');
  const [requiredSkills, setRequiredSkills] = useState<string[]>([]);
  const [preferredSkills, setPreferredSkills] = useState<string[]>([]);
  const [status, setStatus] = useState<JobStatus>('draft');

  // Tag inputs
  const [newRequiredSkill, setNewRequiredSkill] = useState('');
  const [newPreferredSkill, setNewPreferredSkill] = useState('');

  // Validation & UI State
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Initialize or reset form
  useEffect(() => {
    if (jobToEdit) {
      setTitle(jobToEdit.title || '');
      setDepartment(jobToEdit.department || '');
      setLocation(jobToEdit.location || '');
      setEmploymentType((jobToEdit.employment_type as EmploymentType) || 'Full-time');
      setWorkplaceType((jobToEdit.workplace_type as WorkplaceType) || 'On-site');
      setDescription(jobToEdit.description || '');
      setMinimumExperience(
        jobToEdit.minimum_experience !== null && jobToEdit.minimum_experience !== undefined
          ? String(jobToEdit.minimum_experience)
          : ''
      );
      setRequiredSkills(jobToEdit.required_skills ? [...jobToEdit.required_skills] : []);
      setPreferredSkills(jobToEdit.preferred_skills ? [...jobToEdit.preferred_skills] : []);
      setStatus(jobToEdit.status || 'draft');
    } else {
      setTitle('');
      setDepartment('');
      setLocation('');
      setEmploymentType('Full-time');
      setWorkplaceType('On-site');
      setDescription('');
      setMinimumExperience('');
      setRequiredSkills([]);
      setPreferredSkills([]);
      setStatus('draft');
    }
    setErrors({});
    setServerError(null);
    setNewRequiredSkill('');
    setNewPreferredSkill('');
  }, [jobToEdit, isOpen]);

  // Skill chip handlers
  const handleAddRequiredSkill = () => {
    const trimmed = newRequiredSkill.trim();
    if (!trimmed) return;
    if (!requiredSkills.some((s) => s.toLowerCase() === trimmed.toLowerCase())) {
      setRequiredSkills([...requiredSkills, trimmed]);
    }
    setNewRequiredSkill('');
  };

  const handleRemoveRequiredSkill = (skillToRemove: string) => {
    setRequiredSkills(requiredSkills.filter((s) => s !== skillToRemove));
  };

  const handleAddPreferredSkill = () => {
    const trimmed = newPreferredSkill.trim();
    if (!trimmed) return;
    if (!preferredSkills.some((s) => s.toLowerCase() === trimmed.toLowerCase())) {
      setPreferredSkills([...preferredSkills, trimmed]);
    }
    setNewPreferredSkill('');
  };

  const handleRemovePreferredSkill = (skillToRemove: string) => {
    setPreferredSkills(preferredSkills.filter((s) => s !== skillToRemove));
  };

  // Client-side validation
  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!title.trim()) {
      newErrors.title = 'Job title is required';
    } else if (title.trim().length < 2) {
      newErrors.title = 'Job title must be at least 2 characters';
    }

    if (!description.trim()) {
      newErrors.description = 'Job description is required';
    } else if (description.trim().length < 10) {
      newErrors.description = 'Job description must be at least 10 characters long';
    }

    if (minimumExperience.trim() !== '') {
      const expNum = Number(minimumExperience);
      if (isNaN(expNum) || expNum < 0) {
        newErrors.minimumExperience = 'Experience must be a non-negative number';
      } else if (expNum > 50) {
        newErrors.minimumExperience = 'Experience cannot exceed 50 years';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (targetStatus?: JobStatus) => {
    setServerError(null);
    if (!validateForm()) return;

    setIsSubmitting(true);
    const chosenStatus = targetStatus || status;

    const parsedExp =
      minimumExperience.trim() !== '' ? Number(minimumExperience.trim()) : null;

    try {
      if (isEditing && jobToEdit) {
        const updatePayload: JobUpdateInput = {
          title: title.trim(),
          department: department.trim() || null,
          location: location.trim() || null,
          employment_type: employmentType,
          workplace_type: workplaceType,
          description: description.trim(),
          required_skills: requiredSkills,
          preferred_skills: preferredSkills,
          minimum_experience: parsedExp,
          status: chosenStatus,
        };

        const updatedJob = await jobService.updateJob(jobToEdit.id, updatePayload);
        onSuccess(updatedJob, true);
        onClose();
      } else {
        const createPayload: JobCreateInput = {
          title: title.trim(),
          department: department.trim() || null,
          location: location.trim() || null,
          employment_type: employmentType,
          workplace_type: workplaceType,
          description: description.trim(),
          required_skills: requiredSkills,
          preferred_skills: preferredSkills,
          minimum_experience: parsedExp,
          status: chosenStatus,
        };

        const createdJob = await jobService.createJob(createPayload);
        onSuccess(createdJob, false);
        onClose();
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setServerError(err.message);
      } else if (err instanceof Error) {
        setServerError(err.message);
      } else {
        setServerError('Failed to save job role. Please check your connection and try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={isSubmitting ? () => {} : onClose}
      title={isEditing ? 'Edit role' : 'Create a role'}
      description={
        isEditing
          ? 'Update competency criteria, responsibilities, and status.'
          : 'Define role competencies, experience criteria, and responsibilities.'
      }
      maxWidth="700px"
      footer={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClose}
            disabled={isSubmitting}
          >
            Cancel
          </Button>

          <div style={{ display: 'flex', gap: '8px' }}>
            {isEditing ? (
              <>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => handleSubmit(status === 'active' ? 'draft' : 'active')}
                  disabled={isSubmitting}
                >
                  {status === 'active' ? 'Revert to Draft' : 'Publish Role'}
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  onClick={() => handleSubmit(status)}
                  isLoading={isSubmitting}
                >
                  Save changes
                </Button>
              </>
            ) : (
              <>
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => handleSubmit('draft')}
                  disabled={isSubmitting}
                >
                  Save as draft
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  onClick={() => handleSubmit('active')}
                  isLoading={isSubmitting}
                >
                  Publish role
                </Button>
              </>
            )}
          </div>
        </div>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
        style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}
      >
        {/* Server Error Alert */}
        {serverError && (
          <div
            style={{
              padding: '10px 14px',
              backgroundColor: 'var(--color-error-bg)',
              border: '1px solid rgba(195, 68, 68, 0.25)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--color-error)',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
            role="alert"
          >
            <AlertCircle size={16} style={{ flexShrink: 0 }} />
            <span>{serverError}</span>
          </div>
        )}

        {/* Job Title */}
        <Input
          label="Job title *"
          placeholder="e.g. Senior Machine Learning Engineer"
          value={title}
          onChange={(e) => {
            setTitle(e.target.value);
            if (errors.title) setErrors({ ...errors, title: '' });
          }}
          error={errors.title}
          disabled={isSubmitting}
          required
        />

        {/* Department and Location */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <Input
            label="Department"
            placeholder="e.g. Engineering, AI Research"
            value={department}
            onChange={(e) => setDepartment(e.target.value)}
            disabled={isSubmitting}
          />
          <Input
            label="Location"
            placeholder="e.g. San Francisco, CA / London"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            disabled={isSubmitting}
          />
        </div>

        {/* Employment Type & Workplace Type */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Employment type</label>
            <select
              value={employmentType}
              onChange={(e) => setEmploymentType(e.target.value as EmploymentType)}
              disabled={isSubmitting}
              className="form-input"
              style={{ height: '40px', cursor: 'pointer' }}
            >
              <option value="Full-time">Full-time</option>
              <option value="Part-time">Part-time</option>
              <option value="Contract">Contract</option>
              <option value="Internship">Internship</option>
              <option value="Freelance">Freelance</option>
            </select>
          </div>

          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Workplace type</label>
            <select
              value={workplaceType}
              onChange={(e) => setWorkplaceType(e.target.value as WorkplaceType)}
              disabled={isSubmitting}
              className="form-input"
              style={{ height: '40px', cursor: 'pointer' }}
            >
              <option value="On-site">On-site</option>
              <option value="Hybrid">Hybrid</option>
              <option value="Remote">Remote</option>
            </select>
          </div>

          <div className="form-group" style={{ margin: 0 }}>
            <Input
              label="Min experience (years)"
              type="number"
              min="0"
              max="50"
              step="0.5"
              placeholder="e.g. 3"
              value={minimumExperience}
              onChange={(e) => {
                setMinimumExperience(e.target.value);
                if (errors.minimumExperience) {
                  setErrors({ ...errors, minimumExperience: '' });
                }
              }}
              error={errors.minimumExperience}
              disabled={isSubmitting}
            />
          </div>
        </div>

        {/* Required Skills Chip Input */}
        <div className="form-group" style={{ margin: 0 }}>
          <label className="form-label">
            <span>Required competencies</span>
            <span style={{ fontSize: '11.5px', color: 'var(--color-text-secondary)', fontWeight: 400 }}>
              Must-have criteria
            </span>
          </label>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <input
              type="text"
              placeholder="Add required skill (e.g. PyTorch, System Architecture)..."
              value={newRequiredSkill}
              onChange={(e) => setNewRequiredSkill(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleAddRequiredSkill();
                }
              }}
              disabled={isSubmitting}
              className="form-input"
              style={{ flex: 1, height: '36px', fontSize: '13px' }}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={handleAddRequiredSkill}
              disabled={!newRequiredSkill.trim() || isSubmitting}
              leftIcon={<Plus size={14} />}
            >
              Add
            </Button>
          </div>

          {/* Chips list */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', minHeight: '28px' }}>
            {requiredSkills.length === 0 ? (
              <span style={{ fontSize: '12px', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
                No required competencies added yet.
              </span>
            ) : (
              requiredSkills.map((skill) => (
                <span
                  key={skill}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px',
                    padding: '3px 8px',
                    backgroundColor: 'rgba(39, 211, 176, 0.12)',
                    color: '#12343B',
                    border: '1px solid rgba(39, 211, 176, 0.35)',
                    borderRadius: 'var(--radius-xs)',
                    fontSize: '12px',
                    fontWeight: 500,
                  }}
                >
                  {skill}
                  <button
                    type="button"
                    onClick={() => handleRemoveRequiredSkill(skill)}
                    aria-label={`Remove ${skill}`}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#12343B',
                      cursor: 'pointer',
                      padding: 0,
                      display: 'flex',
                      alignItems: 'center',
                    }}
                  >
                    <X size={12} />
                  </button>
                </span>
              ))
            )}
          </div>
        </div>

        {/* Preferred Skills Chip Input */}
        <div className="form-group" style={{ margin: 0 }}>
          <label className="form-label">
            <span>Preferred competencies</span>
            <span style={{ fontSize: '11.5px', color: 'var(--color-text-secondary)', fontWeight: 400 }}>
              Nice-to-have criteria
            </span>
          </label>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <input
              type="text"
              placeholder="Add preferred skill (e.g. Docker, Vector DBs)..."
              value={newPreferredSkill}
              onChange={(e) => setNewPreferredSkill(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleAddPreferredSkill();
                }
              }}
              disabled={isSubmitting}
              className="form-input"
              style={{ flex: 1, height: '36px', fontSize: '13px' }}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={handleAddPreferredSkill}
              disabled={!newPreferredSkill.trim() || isSubmitting}
              leftIcon={<Plus size={14} />}
            >
              Add
            </Button>
          </div>

          {/* Chips list */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', minHeight: '28px' }}>
            {preferredSkills.length === 0 ? (
              <span style={{ fontSize: '12px', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
                No preferred competencies added yet.
              </span>
            ) : (
              preferredSkills.map((skill) => (
                <span
                  key={skill}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px',
                    padding: '3px 8px',
                    backgroundColor: 'var(--color-hover)',
                    color: 'var(--color-text)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-xs)',
                    fontSize: '12px',
                    fontWeight: 500,
                  }}
                >
                  {skill}
                  <button
                    type="button"
                    onClick={() => handleRemovePreferredSkill(skill)}
                    aria-label={`Remove ${skill}`}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--color-text-secondary)',
                      cursor: 'pointer',
                      padding: 0,
                      display: 'flex',
                      alignItems: 'center',
                    }}
                  >
                    <X size={12} />
                  </button>
                </span>
              ))
            )}
          </div>
        </div>

        {/* Job Description */}
        <div className="form-group" style={{ margin: 0 }}>
          <label htmlFor="job-description-input" className="form-label">
            <span>Job description *</span>
            <span style={{ fontSize: '11.5px', color: 'var(--color-text-secondary)', fontWeight: 400 }}>
              Responsibilities and scope (min 10 chars)
            </span>
          </label>
          <textarea
            id="job-description-input"
            rows={5}
            placeholder="Outline the role objectives, expected contributions, team structure, and qualifications..."
            value={description}
            onChange={(e) => {
              setDescription(e.target.value);
              if (errors.description) setErrors({ ...errors, description: '' });
            }}
            disabled={isSubmitting}
            className={`form-input ${errors.description ? 'has-error' : ''}`}
            style={{
              padding: '10px 14px',
              height: 'auto',
              fontFamily: 'inherit',
              lineHeight: 1.5,
              resize: 'vertical',
            }}
            required
          />
          {errors.description && (
            <div className="form-error" role="alert">
              <AlertCircle size={14} />
              <span>{errors.description}</span>
            </div>
          )}
        </div>
      </form>
    </Modal>
  );
};
