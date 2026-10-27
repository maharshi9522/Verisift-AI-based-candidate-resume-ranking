import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Mail, Lock, AlertCircle, ArrowRight } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { AuthLayout } from '../components/auth/AuthLayout';
import { ApiError } from '../services/api';

const loginSchema = z.object({
  email: z.string().min(1, 'Email is required').email('Please enter a valid work email address'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export const Login: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [serverError, setServerError] = useState<string | null>(null);

  const fromLocation = (location.state as any)?.from?.pathname || '/';

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginFormValues) => {
    setServerError(null);
    try {
      await login(data);
      navigate(fromLocation, { replace: true });
    } catch (err: any) {
      if (err instanceof ApiError) {
        setServerError(err.message);
      } else {
        setServerError('Unable to connect to the authentication service. Please verify the backend is running.');
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
          Sign in to Verisift
        </h1>
        <p style={{ fontSize: '13.5px', color: 'var(--color-text-secondary)', margin: 0 }}>
          Enter your recruiter credentials to access your workspace.
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

      <form onSubmit={handleSubmit(onSubmit)} noValidate>
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
          placeholder="Enter your password"
          leftIcon={<Lock size={15} />}
          error={errors.password?.message}
          disabled={isSubmitting}
          autoComplete="current-password"
          {...register('password')}
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
            Sign in
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
        <span>New to Verisift? </span>
        <Link
          to="/signup"
          style={{
            color: 'var(--color-primary)',
            fontWeight: 600,
            textDecoration: 'none',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
          onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
        >
          Create recruiter workspace
        </Link>
      </div>
    </AuthLayout>
  );
};
