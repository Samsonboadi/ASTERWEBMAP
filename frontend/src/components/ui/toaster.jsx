// src/components/ui/toaster.jsx
import React, { useState, useEffect } from 'react';
// Import from toast.jsx, not toast.js
import { Toast, ToastProvider, ToastViewport } from './toast.jsx';

// Create a simple toast context
const ToastContext = React.createContext({
  toast: () => {},
});

export const useToast = () => React.useContext(ToastContext);

// Simple toast implementation with unique ID generation
export const Toaster = () => {
  const [toasts, setToasts] = useState([]);
  const [toastCounter, setToastCounter] = useState(0);

  // Function to add a toast with guaranteed unique ID
  const addToast = ({ title, description, variant = 'default', duration = 5000 }) => {
    const id = `toast_${Date.now()}_${toastCounter}`;
    setToastCounter(prev => prev + 1);
    
    const newToast = { id, title, description, variant, duration };
    setToasts(prev => [...prev, newToast]);
    
    // Auto-remove toast after duration
    setTimeout(() => {
      removeToast(id);
    }, duration);
    
    return id;
  };

  // Function to remove a toast
  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  // Expose the toast function via context and global window
  useEffect(() => {
    window.toast = addToast;
  }, []);

  return (
    <ToastProvider>
      <ToastContext.Provider value={{ toast: addToast }}>
        <ToastViewport>
          {toasts.map(({ id, title, description, variant, duration }) => (
            <Toast 
              key={id}
              variant={variant}
              onMouseDown={() => removeToast(id)}
              style={{ cursor: 'pointer' }}
            >
              {title && <div className="font-medium">{title}</div>}
              {description && <div className="text-sm opacity-90">{description}</div>}
            </Toast>
          ))}
        </ToastViewport>
      </ToastContext.Provider>
    </ToastProvider>
  );
};