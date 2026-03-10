import { useState, useEffect, useMemo, useRef } from 'react'

export default function ModelPicker({ label, value, onChange, models }) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const wrapRef = useRef(null)
  const btnRef = useRef(null)
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 })

  useEffect(() => {
    if (!open) return
    const handler = e => { if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  useEffect(() => {
    if (open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect()
      const spaceBelow = window.innerHeight - r.bottom
      const dropH = 280
      if (spaceBelow < dropH) {
        setPos({ bottom: window.innerHeight - r.top + 4, left: r.left, width: r.width, openUp: true })
      } else {
        setPos({ top: r.bottom + 4, left: r.left, width: r.width, openUp: false })
      }
    }
  }, [open])

  const entries = useMemo(() => {
    if (!models) return []
    if (Array.isArray(models)) {
      return models.map(model => {
        if (typeof model === 'string') {
          return { label: model, value: model, group: 'Other', disabled: false }
        }
        const normalizedValue = model.value ?? model.id ?? model.model_ref
        const normalizedLabel = model.label ?? model.display_name ?? model.value ?? model.id
        const normalizedGroup = model.group ?? model.family ?? model.task ?? 'Other'
        return {
          ...model,
          label: normalizedLabel,
          value: normalizedValue,
          group: normalizedGroup,
          disabled: !!model.disabled || model.enabled === false || model.archived === true,
        }
      }).filter(model => model.value)
    }
    return Object.entries(models).map(([lbl, val]) => ({
      label: lbl,
      value: val,
      group: lbl.match(/^\[([^\]]+)\]/)?.[1] || 'Other',
      disabled: false,
    }))
  }, [models])

  const groups = useMemo(() => {
    const g = {}
    const lc = search.toLowerCase()
    entries.forEach(entry => {
      if (lc && !entry.label.toLowerCase().includes(lc)) return
      const cat = entry.group || 'Other'
      ;(g[cat] = g[cat] || []).push(entry)
    })
    return g
  }, [entries, search])

  const selectedLabel = entries.find(entry => entry.value === value)?.label || value || 'Select model...'
  const displayLabel = selectedLabel.replace(/^\[[^\]]+\]\s*/, '')

  return (
    <div ref={wrapRef}>
      {label && <span className="block text-xs text-slate-400 mb-1">{label}</span>}
      <button ref={btnRef} type="button" onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 hover:border-[var(--accent)] focus:border-[var(--accent)] outline-none transition-colors text-left">
        <span className="truncate">{displayLabel}</span>
        <svg className={`w-3.5 h-3.5 text-slate-400 shrink-0 ml-2 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
      </button>

      {open && (
        <div className="fixed bg-slate-800 border border-slate-600 rounded-lg shadow-2xl overflow-hidden"
          style={{ zIndex: 9999, left: pos.left, width: pos.width, ...(pos.openUp ? { bottom: pos.bottom } : { top: pos.top }) }}>
          <div className="p-2 border-b border-slate-700">
            <input type="text" value={search} onChange={e => setSearch(e.target.value)} placeholder="Search models..."
              autoFocus
              className="w-full bg-slate-700/80 border border-slate-600 rounded-md px-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] outline-none" />
          </div>
          <div className="max-h-56 overflow-y-auto">
            {Object.entries(groups).length === 0 && (
              <div className="px-3 py-3 text-xs text-slate-500 text-center">No models match "{search}"</div>
            )}
            {Object.entries(groups).map(([cat, items]) => (
              <div key={cat}>
                <div className="px-3 py-1 text-[10px] font-bold text-slate-500 uppercase tracking-wider bg-slate-800/60 sticky top-0 z-10">{cat}</div>
                {items.map(m => {
                  const sel = m.value === value
                  return (
                    <div
                      key={m.value}
                      onClick={() => {
                        if (m.disabled) return
                        onChange(m.value)
                        setOpen(false)
                        setSearch('')
                      }}
                      className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors text-xs
                        ${m.disabled
                          ? 'text-slate-600 cursor-not-allowed'
                          : sel
                            ? 'bg-[var(--accent)]/10 text-[var(--accent)]'
                            : 'text-slate-300 hover:bg-slate-700/50'}`}>
                      {sel && <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] shrink-0" />}
                      <span className="truncate">{m.label.replace(/^\[[^\]]+\]\s*/, '')}</span>
                      {m.disabled && <span className="text-[9px] text-slate-600 ml-auto">Disabled</span>}
                      {!m.disabled && <span className="text-[10px] text-slate-600 shrink-0 ml-auto">{m.group || ''}</span>}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
