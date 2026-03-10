export function Card({ children, className = '' }) {
  return <div className={`bg-slate-800/40 border border-slate-700 rounded-xl p-5 ${className}`}>{children}</div>
}

export function SectionLabel({ children }) {
  return <h3 className="text-[11px] font-semibold uppercase tracking-widest text-slate-500 mb-3">{children}</h3>
}

export function Row({ children, className = '' }) {
  return <div className={`flex flex-wrap gap-4 items-end ${className}`}>{children}</div>
}

export function InputGrid({ children, className = '' }) {
  return <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 items-end ${className}`}>{children}</div>
}

export function ParamGrid({ children, className = '' }) {
  return <div className={`grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 items-end ${className}`}>{children}</div>
}

export function Select({ label, value, onChange, options, className = 'flex-1 min-w-[160px]' }) {
  return (
    <div className={className}>
      {label && <span className="block text-xs text-slate-400 mb-1">{label}</span>}
      <select value={value} onChange={e => onChange(e.target.value)}
        className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:border-[var(--accent)] outline-none">
        {options.map(o => typeof o === 'string'
          ? <option key={o} value={o}>{o}</option>
          : <option key={o.value} value={o.value}>{o.label}</option>
        )}
      </select>
    </div>
  )
}

export function TextInput({ label, value, onChange, placeholder = '', className = 'flex-1 min-w-[160px]' }) {
  return (
    <div className={className}>
      {label && <span className="block text-xs text-slate-400 mb-1">{label}</span>}
      <input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:border-[var(--accent)] outline-none" />
    </div>
  )
}

export function RunButton({ onClick, loading, disabled, label = 'Run Attack', loadingLabel = 'Running…' }) {
  return (
    <button onClick={onClick} disabled={loading || disabled}
      className="px-8 py-3 bg-[var(--accent)] hover:brightness-110 text-slate-900 font-bold rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm tracking-wide">
      {loading ? loadingLabel : label}
    </button>
  )
}

export function ErrorMsg({ msg }) {
  if (!msg) return null
  return <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{msg}</p>
}
