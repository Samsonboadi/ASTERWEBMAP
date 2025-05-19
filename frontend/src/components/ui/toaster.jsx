// src/components/ui/toaster.jsx
import React, { useState, useEffect } from 'react';
// Import from toast.jsx, not toast.js
import { Toast, ToastProvider, ToastViewport } from './toast.jsx';

// Create a simple toast context
const ToastContext = React.createContext({
  toast: () => {},
});

export const useToast = () => React.useContext(ToastContext);

// Simple toast implementation
export const Toaster = () => {
  const [toasts, setToasts] = useState([]);

  // Function to add a toast
  const addToast = ({ title, description, variant = 'default', duration = 5000 }) => {
    const id = Date.now().toString();
    setToasts(prev => [...prev, { id, title, description, variant, duration }]);
    return id;
  };

  // Function to remove a toast
  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  // Expose the toast function via context
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