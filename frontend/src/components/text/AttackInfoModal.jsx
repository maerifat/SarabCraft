import { useEffect } from 'react'

const THREAT_COLORS = {
  blackbox: 'border-emerald-500/40 text-emerald-400 bg-emerald-500/10',
  whitebox: 'border-purple-500/40 text-purple-400 bg-purple-500/10',
  score:    'border-amber-500/40  text-amber-400  bg-amber-500/10',
}
const THREAT_LABEL = { blackbox: 'Black-box', whitebox: 'White-box', score: 'Score-based' }

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
            <div className="text-[11px] text-slate-500 mt-0.5">
              {meta.authors || 'Unknown'} · {meta.year || '—'}
              {meta.arxiv && (
                <>
                  {' · '}
                  <a href={`https://arxiv.org/abs/${meta.arxiv}`} target="_blank" rel="noopener noreferrer"
                    className="text-[var(--accent)] hover:underline">
                    arXiv:{meta.arxiv}
                  </a>
                </>
              )}
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-100 text-xl leading-none ml-3 -mt-1 transition-colors">×</button>
        </div>
        
        <div className="flex gap-2 mb-3 flex-wrap">
          <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[meta.threat_model] || THREAT_COLORS.blackbox}`}>
            {THREAT_LABEL[meta.threat_model] || 'Black-box'}
          </span>
          {meta.category && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full border border-slate-600 text-slate-400">
              {meta.category}
            </span>
          )}
        </div>
        
        <p className="text-slate-300 text-[13px] leading-relaxed mb-3">
          {meta.description || 'No description available.'}
        </p>
        
        {meta.paper && (
          <div className="mb-3 p-2 bg-slate-900/50 rounded border border-slate-700">
            <div className="text-[10px] text-slate-500 mb-1">Paper:</div>
            <div className="text-[11px] text-slate-300 italic">{meta.paper}</div>
          </div>
        )}
        
        {meta.params && Object.keys(meta.params).length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700">
            <div className="text-[10px] text-slate-500 mb-1.5 font-medium">Default parameters:</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(meta.params).map(([k, v]) => (
                <span key={k} className="text-[10px] bg-slate-700/60 text-slate-300 px-2 py-1 rounded">
                  {k.replace(/_/g, ' ')}: <span className="text-[var(--accent)] font-medium">{v.default ?? '—'}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
