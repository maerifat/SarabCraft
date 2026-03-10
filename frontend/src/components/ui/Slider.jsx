import { useCallback } from 'react'

export default function Slider({ label, value, onChange, min, max, step, defaultValue, suffix = '' }) {
  const reset = useCallback(() => onChange(defaultValue), [defaultValue, onChange])
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="flex-1 min-w-[140px]">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400 leading-none">{label}</span>
        <div className="flex items-center gap-1">
          <input
            type="number"
            value={value}
            step={step}
            min={min}
            max={max}
            onChange={e => onChange(Number(e.target.value))}
            className="w-16 text-right text-xs bg-transparent text-slate-200 border-b border-slate-600 focus:border-[var(--accent)] outline-none py-0 px-0.5 tabular-nums"
          />
          {suffix && <span className="text-[10px] text-slate-500">{suffix}</span>}
          {defaultValue !== undefined && value !== defaultValue && (
            <button onClick={reset} title="Reset to default" className="ml-0.5 text-slate-500 hover:text-[var(--accent)] transition-colors">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M2 4h2l1-2h6l1 2h2" /><path d="M4 4v9a1 1 0 001 1h6a1 1 0 001-1V4" />
                <path d="M1 8a7 7 0 0113.6-2.3M15 8a7 7 0 01-13.6 2.3" />
                <polyline points="1 3 1 8 6 8" /><polyline points="15 13 15 8 10 8" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div className="relative h-5 flex items-center group">
        <div className="absolute inset-x-0 h-1 bg-slate-700 rounded-full">
          <div className="h-full bg-[var(--accent)]/40 rounded-full transition-all" style={{ width: `${pct}%` }} />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div
          className="absolute w-3 h-3 bg-[var(--accent)] rounded-full shadow-md shadow-[var(--accent)]/30 pointer-events-none transition-all ring-2 ring-[var(--accent)]/20 group-hover:ring-[var(--accent)]/40"
          style={{ left: `calc(${pct}% - 6px)` }}
        />
      </div>
    </div>
  )
}
