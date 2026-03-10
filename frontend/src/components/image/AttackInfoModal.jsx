import { useEffect } from 'react'
import { THREAT_COLORS, THREAT_LABEL } from './attackRegistry'

export default function AttackInfoModal({ name, meta, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose?.()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  if (!name || !meta) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      <div className="relative z-10 w-96 max-w-[90vw] p-5 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl text-xs text-slate-300 leading-relaxed"
        onClick={e => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="font-semibold text-slate-100 text-base">{name}</div>
            <div className="text-[11px] text-slate-500 mt-0.5">{meta.authors || 'Unknown'} · {meta.year || '—'} · {meta.norm || '—'}</div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-100 text-xl leading-none ml-3 -mt-1 transition-colors">×</button>
        </div>
        <div className="flex gap-2 mb-3">
          <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[meta.threat] || THREAT_COLORS.whitebox}`}>{THREAT_LABEL[meta.threat] || 'White-box'}</span>
          {meta.norm && <span className="text-[9px] px-1.5 py-0.5 rounded-full border border-slate-600 text-slate-400">{meta.norm}</span>}
          {meta.cat && <span className="text-[9px] px-1.5 py-0.5 rounded-full border border-slate-600 text-slate-400">{meta.cat}</span>}
        </div>
        <p className="text-slate-300 text-[13px] leading-relaxed">{meta.desc || 'No description available.'}</p>
        {meta.params && Object.values(meta.params).some((param) => !param.hiddenInInfo) && (
          <div className="mt-3 pt-3 border-t border-slate-700">
            <div className="text-[10px] text-slate-500 mb-1.5 font-medium">Default parameters:</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(meta.params).filter(([, v]) => !v.hiddenInInfo).map(([k, v]) => (
                <span key={k} className="text-[10px] bg-slate-700/60 text-slate-300 px-2 py-1 rounded">{v.label || k}: <span className="text-[var(--accent)] font-medium">{v.default ?? '—'}</span></span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
