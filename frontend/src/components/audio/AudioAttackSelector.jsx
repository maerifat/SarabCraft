import { useState, useMemo } from 'react'
import { AUDIO_ATTACK_REGISTRY, THREAT_COLORS, THREAT_LABEL } from './audioAttackRegistry'

export default function AudioAttackSelector({ selected, onSelect }) {
  const [searchTerm, setSearchTerm] = useState('')
  const [infoOpen, setInfoOpen] = useState(null)

  const grouped = useMemo(() => {
    const g = {}
    const lc = searchTerm.toLowerCase()
    Object.values(AUDIO_ATTACK_REGISTRY).forEach(atk => {
      if (lc && !atk.name.toLowerCase().includes(lc) && !atk.cat?.toLowerCase().includes(lc) && !atk.authors?.toLowerCase().includes(lc) && !atk.desc?.toLowerCase().includes(lc))
        return
      const cat = atk.cat || 'Other'
      if (!g[cat]) g[cat] = []
      g[cat].push(atk)
    })
    return g
  }, [searchTerm])

  return (
    <>
      <div className="space-y-2">
        <label className="text-xs text-slate-400 font-medium">Attack Method</label>
        <div className="relative">
          <input type="text" placeholder="Search audio attacks..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
            className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] focus:outline-none" />
          {searchTerm && <button onClick={() => setSearchTerm('')} className="absolute right-2 top-2 text-slate-500 hover:text-slate-300 text-sm">✕</button>}
        </div>

        <div className="max-h-80 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/40 divide-y divide-slate-800/50">
          {Object.entries(grouped).map(([cat, attacks]) => (
            <div key={cat}>
              <div className="px-3 py-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-wider bg-slate-800/40 sticky top-0 z-10">{cat}</div>
              {attacks.map(a => {
                const sel = a.id === selected
                return (
                  <div key={a.id} onClick={() => onSelect(a.id)}
                    className={`flex items-center gap-2 px-3 py-2.5 cursor-pointer transition-colors ${sel ? 'bg-[var(--accent)]/10 border-l-2 border-[var(--accent)]' : 'hover:bg-slate-800/50 border-l-2 border-transparent'}`}>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`text-sm truncate ${sel ? 'text-[var(--accent)] font-semibold' : 'text-slate-200'}`}>{a.name}</span>
                        <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[a.threat] || THREAT_COLORS.whitebox}`}>
                          {THREAT_LABEL[a.threat] || 'White-box'}
                        </span>
                        <span className="text-[10px] text-slate-600">{a.year}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">{a.authors} — {a.norm}</div>
                    </div>
                    <button onClick={e => { e.stopPropagation(); setInfoOpen(infoOpen === a.id ? null : a.id) }}
                      className="flex-shrink-0 w-5 h-5 rounded-full bg-slate-700/60 hover:bg-[var(--accent)]/30 flex items-center justify-center text-[10px] text-slate-400 hover:text-[var(--accent)] transition-colors font-serif italic" title="Info">
                      i
                    </button>
                  </div>
                )
              })}
            </div>
          ))}
          {Object.keys(grouped).length === 0 && <div className="px-3 py-4 text-xs text-slate-500 text-center">No attacks match "{searchTerm}"</div>}
        </div>
      </div>

      {infoOpen && AUDIO_ATTACK_REGISTRY[infoOpen] && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={() => setInfoOpen(null)}>
          <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
          <div className="relative z-10 w-96 max-w-[90vw] p-5 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl text-xs text-slate-300 leading-relaxed"
            onClick={e => e.stopPropagation()}>
            {(() => {
              const meta = AUDIO_ATTACK_REGISTRY[infoOpen]
              return (<>
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="font-semibold text-slate-100 text-base">{meta.name}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">{meta.authors} · {meta.year} · {meta.norm}</div>
                  </div>
                  <button onClick={() => setInfoOpen(null)} className="text-slate-400 hover:text-slate-100 text-xl leading-none ml-3 -mt-1 transition-colors">×</button>
                </div>
                <div className="flex gap-2 mb-3">
                  <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[meta.threat] || THREAT_COLORS.whitebox}`}>{THREAT_LABEL[meta.threat] || 'White-box'}</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full border border-slate-600 text-slate-400">{meta.norm}</span>
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full border border-slate-600 text-slate-400">{meta.cat}</span>
                </div>
                <p className="text-slate-300 text-[13px] leading-relaxed">{meta.desc}</p>
              </>)
            })()}
          </div>
        </div>
      )}
    </>
  )
}
