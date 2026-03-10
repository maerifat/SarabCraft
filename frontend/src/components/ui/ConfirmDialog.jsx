import { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react'

const ConfirmContext = createContext(null)

const VARIANT_STYLES = {
  danger: 'bg-red-600 hover:bg-red-500 text-white',
  warning: 'bg-amber-600 hover:bg-amber-500 text-white',
  default: 'bg-cyan-600 hover:bg-cyan-500 text-white',
}

function Dialog({ config, onResolve }) {
  const confirmRef = useRef(null)
  const {
    title = 'Confirm',
    message = 'Are you sure?',
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    variant = 'default',
  } = config

  useEffect(() => {
    confirmRef.current?.focus()
  }, [])

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onResolve(false)
      if (e.key === 'Tab') {
        const focusable = document.querySelectorAll('[data-confirm-focusable]')
        if (focusable.length === 0) return
        const first = focusable[0]
        const last = focusable[focusable.length - 1]
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onResolve])

  return (
    <div
      className="fixed inset-0 z-[9998] flex items-center justify-center"
      onClick={() => onResolve(false)}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative bg-slate-800 border border-slate-700/60 rounded-xl shadow-2xl p-6 max-w-md w-full mx-4 animate-fade-in"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-message"
      >
        <h2 id="confirm-title" className="text-lg font-semibold text-slate-100 mb-2">
          {title}
        </h2>
        <p id="confirm-message" className="text-sm text-slate-400 leading-relaxed mb-6">
          {message}
        </p>
        <div className="flex items-center justify-end gap-3">
          <button
            data-confirm-focusable
            onClick={() => onResolve(false)}
            className="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 border border-slate-600 text-slate-300 text-sm font-medium transition-colors"
          >
            {cancelText}
          </button>
          <button
            ref={confirmRef}
            data-confirm-focusable
            onClick={() => onResolve(true)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${VARIANT_STYLES[variant] || VARIANT_STYLES.default}`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}

export function ConfirmProvider({ children }) {
  const [pending, setPending] = useState(null)

  const confirm = useCallback((options = {}) => {
    return new Promise((resolve) => {
      setPending({ ...options, resolve })
    })
  }, [])

  const handleResolve = useCallback((value) => {
    if (pending?.resolve) pending.resolve(value)
    setPending(null)
  }, [pending])

  return (
    <ConfirmContext.Provider value={{ confirm }}>
      {children}
      {pending && <Dialog config={pending} onResolve={handleResolve} />}
    </ConfirmContext.Provider>
  )
}

export function useConfirm() {
  const ctx = useContext(ConfirmContext)
  if (!ctx) throw new Error('useConfirm must be used within a ConfirmProvider')
  return ctx
}
