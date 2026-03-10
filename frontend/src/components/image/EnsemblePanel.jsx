import { useState } from 'react'
import { Select } from '../ui/Section'
import { SARABCRAFT_R1_NAME } from './sarabcraftR1'

export default function EnsemblePanel({ attack, modelOpts, model, ensembleModels, setEnsembleModels, ensembleMode, setEnsembleMode }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mt-4 pt-3 border-t border-slate-700/50">
      <div className="flex items-center gap-3 mb-2">
        <button onClick={() => setOpen(o => !o)}
          className="flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-[var(--accent)] transition-colors">
          <svg className={`w-3 h-3 transition-transform ${open ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20"><path d="M6 4l8 6-8 6V4z"/></svg>
          Ensemble Attack {ensembleModels.length > 0 && <span className="text-[10px] bg-[var(--accent)]/20 text-[var(--accent)] px-1.5 py-0.5 rounded-full">{ensembleModels.length} selected</span>}
        </button>
        {ensembleModels.length > 0 && <button onClick={() => setEnsembleModels([])} className="text-[10px] text-slate-500 hover:text-red-400">Clear</button>}
      </div>
      {open && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-1.5 max-h-48 overflow-y-auto p-2 bg-slate-900/50 rounded-lg border border-slate-700/50">
            {modelOpts.filter(o => o.value !== model).map(o => {
              const checked = ensembleModels.includes(o.value)
              return (
                <label key={o.value} className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-xs transition-colors ${checked ? 'bg-[var(--accent)]/10 text-[var(--accent)]' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'}`}>
                  <input type="checkbox" checked={checked}
                    onChange={() => setEnsembleModels(prev => checked ? prev.filter(v => v !== o.value) : [...prev, o.value])}
                    className="accent-[var(--accent)] w-3 h-3 flex-shrink-0" />
                  <span className="truncate">{o.label}</span>
                </label>
              )
            })}
          </div>
          {ensembleModels.length > 0 && attack === SARABCRAFT_R1_NAME && <Select label="Ensemble mode" value={ensembleMode} onChange={setEnsembleMode} options={['Simultaneous', 'Alternating']} className="max-w-xs" />}
          {ensembleModels.length > 0 && attack !== SARABCRAFT_R1_NAME && <div className="text-[10px] text-slate-500 italic">Ensemble uses logit averaging (simultaneous). Alternating mode is available only with SarabCraft R1.</div>}
        </div>
      )}
    </div>
  )
}
