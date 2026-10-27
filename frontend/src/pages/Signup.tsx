import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Link, useNavigate } from 'react-router-dom';
import { User as UserIcon, Mail, Lock, AlertCircle, ArrowRight, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { AuthLayout } from '../components/auth/AuthLayout';
import { ApiError } from '../services/api';

const signupSchema = z
  .object({
    full_name: z
      .string()
      .min(2, 'Full name must be at least 2 characters')
      .max(80, 'Full name cannot exceed 80 characters'),
    email: z.string().min(1, 'Work email is required').email('Please enter a valid work email address'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .max(100, 'Password cannot exceed 100 characters'),
    confirm_password: z.string().min(1, 'Please confirm your password'),
  })
  .refine((data) => data.password === data.confirm_password, {
    message: 'Passwords do not match',
    path: ['confirm_password'],
  });

type SignupFormValues = z.infer<typeof signupSchema>;

export const Signup: React.FC = () => {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [serverError, setServerError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<SignupFormValues>({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      full_name: '',
      email: '',
      password: '',
      confirm_password: '',
    },
  });

  const onSubmit = async (data: SignupFormValues) => {
    setServerError(null);
    setSuccessMessage(null);
    try {
      await signup({
        full_name: data.full_name,
        email: data.email,
        password: data.password,
        confirm_password: data.confirm_password,
      });
      setSuccessMessage('Workspace created successfully. Redirecting...');
      setTimeout(() => {
        navigate('/', { replace: true });
      }, 400);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setServerError(err.message);
      } else {
        setServerError('Unable to connect to the registration service. Please verify the backend is running.');
      }
    }
  };

  return (
    <AuthLayout>
      <div style={{ marginBottom: '22px' }}>
        <h1
          style={{
            fontFamily: "var(--font-heading, 'Sora', sans-serif)",
            fontSize: '22px',
            fontWeight: 600,
            color: 'var(--color-text)',
            letterSpacing: '-0.02em',
            marginBottom: '4px',
          }}
        >
          Create workspace
        </h1>
        <p style={{ fontSize: '13.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
          Set up your recruitment intelligence account.
        </p>
      </div>

      {serverError && (
        <div
          style={{
            backgroundColor: 'var(--color-error-bg)',
            border: '1px solid rgba(195, 68, 68, 0.25)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 12px',
            marginBottom: '18px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--color-error)',
            fontSize: '12.5px',
            lineHeight: 1.4,
          }}
          role="alert"
        >
          <AlertCircle size={15} style={{ flexShrink: 0 }} />
          <div>{serverError}</div>
        </div>
      )}

      {successMessage && (
        <div
          style={{
            backgroundColor: 'var(--color-success-bg)',
            border: '1px solid rgba(22, 134, 106, 0.25)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 12px',
            marginBottom: '18px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: 'var(--color-success)',
            fontSize: '12.5px',
          }}
          role="status"
        >
          <CheckCircle2 size={15} style={{ flexShrink: 0 }} />
          <div>{successMessage}</div>
        </div>
      )}

      <form onSubmit={handleSubmit(onSubmit)} noValidate>
        <Input
          label="Full name"
          type="text"
          placeholder="e.g. Alex Henderson"
          leftIcon={<UserIcon size={15} />}
          error={errors.full_name?.message}
          disabled={isSubmitting}
          autoComplete="name"
          {...register('full_name')}
        />

        <Input
          label="Work email"
          type="email"
          placeholder="recruiter@company.com"
          leftIcon={<Mail size={15} />}
          error={errors.email?.message}
          disabled={isSubmitting}
          autoComplete="email"
          {...register('email')}
        />

        <Input
          label="Password"
          isPassword
          placeholder="Minimum 8 characters"
          leftIcon={<Lock size={15} />}
          error={errors.password?.message}
          helperText="At least 8 characters"
          disabled={isSubmitting}
          autoComplete="new-password"
          {...register('password')}
        />

        <Input
          label="Confirm password"
          isPassword
          placeholder="Re-enter password"
          leftIcon={<Lock size={15} />}
          error={errors.confirm_password?.message}
          disabled={isSubmitting}
          autoComplete="new-password"
          {...register('confirm_password')}
        />

        <div style={{ marginTop: '22px' }}>
          <Button
            type="submit"
            variant="primary"
            size="lg"
            isLoading={isSubmitting}
            rightIcon={!isSubmitting ? <ArrowRight size={15} /> : undefined}
            style={{ width: '100%' }}
          >
            Create workspace
          </Button>
        </div>
      </form>

      <div
        style={{
          marginTop: '22px',
          paddingTop: '18px',
          borderTop: '1px solid var(--color-border-subtle)',
          textAlign: 'center',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
        }}
      >
        <span>Already have an account? </span>
        <Link
          to="/login"
          style={{
            color: 'var(--color-primary)',
            fontWeight: 600,
            textDecoration: 'none',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
          onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
        >
          Sign in
        </Link>
      </div>
    </AuthLayout>
  );
};
