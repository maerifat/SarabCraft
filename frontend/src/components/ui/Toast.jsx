import { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react'

const ToastContext = createContext(null)

const MAX_TOASTS = 3

const TYPE_STYLES = {
  success: {
    border: 'border-emerald-500/40',
    icon: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
  },
  error: {
    border: 'border-red-500/40',
    icon: 'text-red-400',
    bg: 'bg-red-500/10',
  },
  info: {
    border: 'border-cyan-500/40',
    icon: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
  },
}

const ICONS = {
  success: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  error: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
    </svg>
  ),
  info: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
    </svg>
  ),
}

function ToastItem({ toast, onDismiss }) {
  const [exiting, setExiting] = useState(false)
  const timerRef = useRef(null)
  const style = TYPE_STYLES[toast.type] || TYPE_STYLES.info

  const dismiss = useCallback(() => {
    setExiting(true)
    setTimeout(() => onDismiss(toast.id), 300)
  }, [onDismiss, toast.id])

  useEffect(() => {
    timerRef.current = setTimeout(dismiss, toast.duration)
    return () => clearTimeout(timerRef.current)
  }, [dismiss, toast.duration])

  return (
    <div
      role="alert"
      className={`
        flex items-start gap-3 px-4 py-3 rounded-lg border backdrop-blur-sm shadow-xl
        bg-slate-800/95 ${style.border}
        ${exiting ? 'animate-slide-out-right' : 'animate-slide-in-right'}
      `}
      style={{ minWidth: 300, maxWidth: 420 }}
    >
      <div className={`shrink-0 mt-0.5 ${style.icon}`}>
        {ICONS[toast.type]}
      </div>
      <p className="text-sm text-slate-200 leading-relaxed flex-1">{toast.message}</p>
      <button
        onClick={dismiss}
        className="shrink-0 text-slate-500 hover:text-slate-300 transition-colors mt-0.5"
        aria-label="Dismiss"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}

let toastCounter = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback(({ type = 'info', message, duration = 4000 }) => {
    const id = ++toastCounter
    setToasts((prev) => {
      const next = [...prev, { id, type, message, duration }]
      if (next.length > MAX_TOASTS) {
        return next.slice(next.length - MAX_TOASTS)
      }
      return next
    })
    return id
  }, [])

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div
        className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2"
        aria-live="polite"
      >
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onDismiss={removeToast} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within a ToastProvider')
  return ctx
}
