import React, { useState, forwardRef } from 'react';
import { Eye, EyeOff, AlertCircle } from 'lucide-react';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  isPassword?: boolean;
  leftIcon?: React.ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(({
  label,
  error,
  helperText,
  isPassword = false,
  leftIcon,
  type = 'text',
  id,
  className = '',
  disabled,
  ...props
}, ref) => {
  const [showPassword, setShowPassword] = useState(false);
  const inputId = id || props.name || Math.random().toString(36).substring(7);

  const resolvedType = isPassword ? (showPassword ? 'text' : 'password') : type;

  return (
    <div className="form-group">
      {label && (
        <label htmlFor={inputId} className="form-label">
          {label}
        </label>
      )}

      <div className="form-input-wrapper">
        {leftIcon && (
          <div
            style={{
              position: 'absolute',
              left: '12px',
              display: 'flex',
              alignItems: 'center',
              color: 'var(--color-text-secondary)',
              pointerEvents: 'none',
            }}
          >
            {leftIcon}
          </div>
        )}

        <input
          ref={ref}
          id={inputId}
          type={resolvedType}
          disabled={disabled}
          className={`form-input ${error ? 'has-error' : ''} ${className}`.trim()}
          style={{
            paddingLeft: leftIcon ? '40px' : '14px',
            paddingRight: isPassword ? '44px' : '14px',
            opacity: disabled ? 0.6 : 1,
            cursor: disabled ? 'not-allowed' : 'text',
          }}
          aria-invalid={!!error}
          aria-describedby={error ? `${inputId}-error` : helperText ? `${inputId}-helper` : undefined}
          {...props}
        />

        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            tabIndex={-1}
            aria-label={showPassword ? 'Hide password' : 'Show password'}
            style={{
              position: 'absolute',
              right: '10px',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--color-text-secondary)',
              display: 'flex',
              alignItems: 'center',
              padding: '6px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        )}
      </div>

      {error ? (
        <div id={`${inputId}-error`} className="form-error" role="alert">
          <AlertCircle size={14} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      ) : helperText ? (
        <div id={`${inputId}-helper`} className="form-helper">
          {helperText}
        </div>
      ) : null}
    </div>
  );
});

Input.displayName = 'Input';
