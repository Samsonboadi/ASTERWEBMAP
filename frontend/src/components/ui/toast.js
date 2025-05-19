// src/components/ui/toast.js
// This is a simple toast utility to make it easier to show toasts
// This will be imported by components using "...components/ui/toast"

// Simple implementation - in a real app this would use a proper toast library
export const toast = (options) => {
  if (typeof window !== 'undefined' && window.toast) {
    return window.toast(options);
  }
  
  // Fallback to console if toast is not available
  console.log('Toast:', options);
};

// Do not export Toast component from here