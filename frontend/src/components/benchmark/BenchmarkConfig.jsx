import { useState, useEffect, useMemo, useRef } from 'react'
import { Card, SectionLabel, Select, RunButton } from '../ui/Section'
import ModelPicker from '../ui/ModelPicker'
import { SARABCRAFT_R1_NAME } from '../image/sarabcraftR1'

const UNTARGETED_AUDIO_BENCHMARK_ATTACKS = new Set(['Speech Jamming'])

function AttackChecklist({ groups, selected, onChange }) {
  const toggle = name => {
    onChange(prev => prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name])
  }
  const allNames = useMemo(() => {
    if (Array.isArray(groups)) return groups.map(a => a.name)
    return Object.values(groups).flat().map(a => a.name)
  }, [groups])

  const isCatSelected = cat => {
    const names = groups[cat].map(a => a.name)
    return names.length > 0 && names.every(n => selected.includes(n))
  }

  const selectCategory = cat => {
    const names = groups[cat].map(a => a.name)
    onChange(prev => isCatSelected(cat)
      ? prev.filter(n => !names.includes(n))
      : [...new Set([...prev, ...names])]
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5 mb-2">
        <SmallBtn active={selected.length === allNames.length && allNames.length > 0} onClick={() => onChange(selected.length === allNames.length ? [] : allNames)}>Select All</SmallBtn>
        <SmallBtn active={selected.length === 0} onClick={() => onChange([])}>Clear</SmallBtn>
        {!Array.isArray(groups) && Object.keys(groups).map(cat => (
          <SmallBtn key={cat} active={isCatSelected(cat)} onClick={() => selectCategory(cat)}>{cat}</SmallBtn>
        ))}
      </div>

      <div className="max-h-56 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/30 divide-y divide-slate-800/40">
        {Array.isArray(groups) ? (
          groups.map(a => (
            <AttackItem key={a.name} a={a} checked={selected.includes(a.name)} onToggle={toggle} />
          ))
        ) : (
          Object.entries(groups).map(([cat, attacks]) => (
            <div key={cat}>
              <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 bg-slate-800/30 sticky top-0 z-[1]">{cat}</div>
              {attacks.map(a => (
                <AttackItem key={a.name} a={a} checked={selected.includes(a.name)} onToggle={toggle} />
              ))}
            </div>
          ))
        )}
      </div>
      <p className="text-[10px] text-slate-500">{selected.length} attack{selected.length !== 1 ? 's' : ''} selected</p>
    </div>
  )
}

function AttackItem({ a, checked, onToggle }) {
  return (
    <label className={`flex items-center gap-2.5 px-3 py-2 cursor-pointer transition-colors text-xs
      ${checked ? 'bg-[var(--accent)]/8 text-[var(--accent-text,theme(colors.cyan.300))]' : 'text-slate-400 hover:bg-slate-800/40'}`}>
      <input type="checkbox" checked={checked} onChange={() => onToggle(a.name)}
        className="w-3.5 h-3.5 rounded accent-[var(--accent)] shrink-0" />
      <span className="truncate flex-1">{a.name}</span>
      {a.year && <span className="text-[10px] text-slate-600 shrink-0">{a.year}</span>}
    </label>
  )
}

function SmallBtn({ children, active, onClick }) {
  return (
    <button onClick={onClick} className={`text-[10px] px-2 py-0.5 rounded border transition
      ${active
        ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20 hover:bg-[var(--accent)]/20'
        : 'bg-slate-700/40 text-slate-400 border-slate-600/30 hover:text-[var(--accent)] hover:bg-slate-700/60'
      }`}>
      {children}
    </button>
  )
}

function ParamModePanel({ domain, mode, onModeChange, preset, onPresetChange, sweep, onSweepChange }) {
  const presetLabels = { conservative: 'Conservative', balanced: 'Balanced', aggressive: 'Aggressive' }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        {['preset', 'sweep'].map(m => (
          <button key={m} onClick={() => onModeChange(m)}
            className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition border
              ${mode === m
                ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20'
                : 'bg-slate-800/40 text-slate-400 border-slate-600/30 hover:bg-slate-800/60'}`}>
            {m === 'preset' ? 'Presets' : 'Parameter Sweep'}
          </button>
        ))}
      </div>

      {mode === 'preset' && (
        <div className="grid grid-cols-3 gap-2">
          {Object.entries(presetLabels).map(([key, label]) => (
            <button key={key} onClick={() => onPresetChange(key)}
              className={`px-3 py-3 rounded-lg text-xs font-medium transition border text-center
                ${preset === key
                  ? 'bg-[var(--accent)]/15 text-[var(--accent)] border-[var(--accent)]/30'
                  : 'bg-slate-800/40 text-slate-400 border-slate-600/30 hover:bg-slate-800/60'}`}>
              <span className="block font-semibold">{label}</span>
              <span className="block text-[10px] text-slate-500 mt-0.5">
                {domain === 'image'
                  ? key === 'conservative' ? 'ε=4, 20 iters' : key === 'balanced' ? 'ε=16, 50 iters' : 'ε=32, 100 iters'
                  : key === 'conservative' ? 'ε=0.02, 100 iters' : key === 'balanced' ? 'ε=0.05, 300 iters' : 'ε=0.1, 500 iters'
                }
              </span>
            </button>
          ))}
        </div>
      )}

      {mode === 'sweep' && (
        <div className="space-y-3">
          <SweepInput label={domain === 'image' ? 'Epsilon (/255)' : 'Epsilon'}
            value={sweep.epsilon || ''} onChange={v => onSweepChange({ ...sweep, epsilon: v })}
            placeholder={domain === 'image' ? '4, 8, 16, 32' : '0.02, 0.05, 0.1'} />
          <SweepInput label="Iterations"
            value={sweep.iterations || ''} onChange={v => onSweepChange({ ...sweep, iterations: v })}
            placeholder="20, 50, 100" />
          {sweep.epsilon && sweep.iterations && (
            <p className="text-[10px] text-amber-400/80">
              {_parseSweepNums(sweep.epsilon).length * _parseSweepNums(sweep.iterations).length} param combinations per attack
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function SweepInput({ label, value, onChange, placeholder }) {
  return (
    <div>
      <span className="block text-xs text-slate-400 mb-1">{label} <span className="text-slate-600">(comma-separated)</span></span>
      <input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:border-[var(--accent)] outline-none" />
    </div>
  )
}

function _parseSweepNums(str) {
  if (!str) return []
  return str.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
}

function TransferTargetPicker({ domain, targets, onChange, imageLocalTargets, imageRemoteTargets, audioRemoteTargets }) {
  const localModelIds = targets.local_model_ids || []
  const remoteTargetIds = targets.remote_target_ids || []

  const toggleLocalModel = val => {
    const next = localModelIds.includes(val) ? localModelIds.filter(m => m !== val) : [...localModelIds, val]
    onChange({ ...targets, local_model_ids: next })
  }

  const toggleRemoteTarget = val => {
    const next = remoteTargetIds.includes(val) ? remoteTargetIds.filter(id => id !== val) : [...remoteTargetIds, val]
    onChange({ ...targets, remote_target_ids: next })
  }

  if (domain === 'image') {
    const modelEntries = Array.isArray(imageLocalTargets) ? imageLocalTargets : []
    const remoteEntries = Array.isArray(imageRemoteTargets) ? imageRemoteTargets : []
    const groups = {}
    modelEntries.forEach(item => {
      const cat = item.group || item.label?.match(/^\[([^\]]+)\]/)?.[1] || 'Other'
      ;(groups[cat] = groups[cat] || []).push(item)
    })

    const isCatSelected = cat => {
      const vals = groups[cat].map(m => m.value)
      return vals.length > 0 && vals.every(v => localModelIds.includes(v))
    }

    return (
      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between flex-wrap gap-1.5 mb-2">
            <span className="text-xs text-slate-400">{localModelIds.length} local transfer model{localModelIds.length !== 1 ? 's' : ''} selected</span>
            <div className="flex flex-wrap gap-1.5">
              {Object.keys(groups).map(cat => (
                <SmallBtn key={cat} active={isCatSelected(cat)} onClick={() => {
                  const vals = groups[cat].map(m => m.value)
                  if (isCatSelected(cat)) {
                    onChange({ ...targets, local_model_ids: localModelIds.filter(m => !vals.includes(m)) })
                  } else {
                    onChange({ ...targets, local_model_ids: [...new Set([...localModelIds, ...vals])] })
                  }
                }}>{cat}</SmallBtn>
              ))}
              <SmallBtn active={localModelIds.length === modelEntries.length && modelEntries.length > 0} onClick={() => {
                const allVals = modelEntries.map(item => item.value)
                onChange({ ...targets, local_model_ids: localModelIds.length === modelEntries.length ? [] : allVals })
              }}>All</SmallBtn>
              <SmallBtn active={localModelIds.length === 0} onClick={() => onChange({ ...targets, local_model_ids: [] })}>Clear</SmallBtn>
            </div>
          </div>
          <div className="max-h-40 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/30 divide-y divide-slate-800/40">
            {modelEntries.map(item => (
              <label key={item.value} className={`flex items-center gap-2.5 px-3 py-1.5 cursor-pointer text-xs transition-colors
                ${localModelIds.includes(item.value)
                  ? 'bg-[var(--accent)]/8 text-[var(--accent-text,theme(colors.cyan.300))]'
                  : 'text-slate-400 hover:bg-slate-800/40'}`}>
                <input type="checkbox" checked={localModelIds.includes(item.value)} onChange={() => toggleLocalModel(item.value)}
                  className="w-3.5 h-3.5 rounded accent-[var(--accent)] shrink-0" />
                <span className="truncate">{(item.label || item.value).replace(/^\[[^\]]+\]\s*/, '')}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between flex-wrap gap-1.5 mb-2">
            <span className="text-xs text-slate-400">{remoteTargetIds.length} remote verification target{remoteTargetIds.length !== 1 ? 's' : ''} selected</span>
            <div className="flex flex-wrap gap-1.5">
              <SmallBtn active={remoteTargetIds.length === remoteEntries.length && remoteEntries.length > 0} onClick={() => {
                const allVals = remoteEntries.map(item => item.value)
                onChange({ ...targets, remote_target_ids: remoteTargetIds.length === remoteEntries.length ? [] : allVals })
              }}>All</SmallBtn>
              <SmallBtn active={remoteTargetIds.length === 0} onClick={() => onChange({ ...targets, remote_target_ids: [] })}>Clear</SmallBtn>
            </div>
          </div>
          <div className="max-h-40 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/30 divide-y divide-slate-800/40">
            {remoteEntries.map(item => (
              <label key={item.value} className={`flex items-center gap-2.5 px-3 py-1.5 cursor-pointer text-xs transition-colors
                ${remoteTargetIds.includes(item.value)
                  ? 'bg-[var(--accent)]/8 text-[var(--accent-text,theme(colors.cyan.300))]'
                  : 'text-slate-400 hover:bg-slate-800/40'}`}>
                <input type="checkbox" checked={remoteTargetIds.includes(item.value)} onChange={() => toggleRemoteTarget(item.value)}
                  className="w-3.5 h-3.5 rounded accent-[var(--accent)] shrink-0" />
                <span className="truncate">{item.label || item.value}</span>
                {item.backend && <span className="ml-auto text-[10px] text-slate-600">{item.backend}</span>}
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">Preprocess:</span>
          <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
            <input type="radio" name="pp" checked={targets.preprocess_mode === 'exact'}
              onChange={() => onChange({ ...targets, preprocess_mode: 'exact' })} className="accent-[var(--accent)]" />
            Exact (pixel-perfect)
          </label>
          <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
            <input type="radio" name="pp" checked={targets.preprocess_mode === 'standard'}
              onChange={() => onChange({ ...targets, preprocess_mode: 'standard' })} className="accent-[var(--accent)]" />
            Standard (real-world)
          </label>
        </div>
      </div>
    )
  }

  const audioTargets = Array.isArray(audioRemoteTargets) ? audioRemoteTargets : []

  return (
    <div className="space-y-2">
      {audioTargets.map(item => (
        <label key={item.value} className={`flex items-center gap-2.5 px-3 py-2 rounded-lg cursor-pointer text-xs transition border
          ${remoteTargetIds.includes(item.value)
            ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20'
            : 'text-slate-400 border-slate-600/30 hover:bg-slate-800/40'}`}>
          <input type="checkbox" checked={remoteTargetIds.includes(item.value)} onChange={() => toggleRemoteTarget(item.value)}
            className="w-3.5 h-3.5 rounded accent-[var(--accent)]" />
          <span className="truncate flex-1">{item.label || item.value}</span>
          {item.backend && <span className="text-[10px] text-slate-600 shrink-0">{item.backend}</span>}
        </label>
      ))}
      <p className="text-[10px] text-slate-500">Cloud services require configured credentials (Settings → Credentials)</p>
    </div>
  )
}

function ImgUpload({ label, preview, onChange }) {
  return (
    <div>
      <span className="block text-xs text-slate-400 mb-1">{label}</span>
      <label className="block aspect-square border-2 border-dashed border-slate-600 hover:border-[var(--accent)]/50 rounded-lg cursor-pointer overflow-hidden bg-slate-800/60 transition relative">
        {preview
          ? <img src={preview} alt="" className="w-full h-full object-cover" />
          : <div className="flex items-center justify-center h-full text-slate-500 text-xs">Click to upload</div>
        }
        <input type="file" accept="image/*" onChange={onChange} className="absolute inset-0 opacity-0 cursor-pointer" />
      </label>
    </div>
  )
}

export default function BenchmarkConfig({
  domain, setDomain,
  imageAttackGroups, audioAttacks, allModels, asrModels,
  imageLocalTargets, imageRemoteTargets, audioRemoteTargets,
  onRun, loading,
}) {
  const [inputFile, setInputFile] = useState(null)
  const [inputPreview, setInputPreview] = useState(null)
  const [targetFile, setTargetFile] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [audioName, setAudioName] = useState('')
  const [targetText, setTargetText] = useState('')
  const [sourceModel, setSourceModel] = useState('')

  const [selectedAttacks, setSelectedAttacks] = useState([])

  const [paramMode, setParamMode] = useState('preset')
  const [preset, setPreset] = useState('balanced')
  const [sweep, setSweep] = useState({ epsilon: '', iterations: '' })

  const [transferTargets, setTransferTargets] = useState({ local_model_ids: [], preprocess_mode: 'exact', remote_target_ids: [] })

  useEffect(() => {
    setSourceModel('')
    setSelectedAttacks([])
  }, [domain])

  useEffect(() => {
    if (domain === 'image' && allModels?.length && !sourceModel) {
      setSourceModel(allModels[0]?.value || '')
    }
    if (domain === 'audio' && asrModels && asrModels.length && !sourceModel) {
      setSourceModel(asrModels[0]?.value || asrModels[0] || '')
    }
  }, [domain, allModels, asrModels, sourceModel])

  useEffect(() => {
    if (domain === 'image' && imageLocalTargets?.length && transferTargets.local_model_ids.length === 0) {
      const first3 = imageLocalTargets.slice(0, 3).map(item => item.value)
      setTransferTargets(t => ({ ...t, local_model_ids: first3 }))
    }
  }, [domain, imageLocalTargets, transferTargets.local_model_ids.length])

  const pickFile = (setter, previewSetter) => e => {
    const f = e.target.files?.[0]
    setter(f)
    if (f) { const r = new FileReader(); r.onload = ev => previewSetter(ev.target.result); r.readAsDataURL(f) }
  }
  const pickAudio = e => {
    const f = e.target.files?.[0]
    setAudioFile(f)
    setAudioName(f?.name || '')
  }

  const totalCombos = useMemo(() => {
    if (!selectedAttacks.length) return 0
    if (paramMode === 'preset') return selectedAttacks.length
    const eps = _parseSweepNums(sweep.epsilon)
    const iters = _parseSweepNums(sweep.iterations)
    return selectedAttacks.length * Math.max(eps.length, 1) * Math.max(iters.length, 1)
  }, [selectedAttacks, paramMode, sweep])

  const audioTargetRequired = useMemo(() => (
    domain === 'audio' && selectedAttacks.some(name => !UNTARGETED_AUDIO_BENCHMARK_ATTACKS.has(name))
  ), [domain, selectedAttacks])

  const handleRun = () => {
    const fd = new FormData()
    fd.append('domain', domain)
    fd.append('attacks_json', JSON.stringify(selectedAttacks))
    fd.append('param_mode', paramMode)
    fd.append('param_preset', preset)
    fd.append('source_model', sourceModel)
    fd.append('preprocess_mode', transferTargets.preprocess_mode || 'exact')
    fd.append('transfer_targets_json', JSON.stringify(transferTargets))

    if (paramMode === 'sweep') {
      const cfg = {}
      const eps = _parseSweepNums(sweep.epsilon)
      const iters = _parseSweepNums(sweep.iterations)
      if (eps.length) cfg.epsilon = eps
      if (iters.length) cfg.iterations = iters
      fd.append('param_sweep_json', JSON.stringify(cfg))
    }

    if (domain === 'image') {
      if (!inputFile || !targetFile) return
      fd.append('input_file', inputFile)
      fd.append('target_file', targetFile)
    } else {
      if (!audioFile || (audioTargetRequired && !targetText.trim())) return
      fd.append('input_file', audioFile)
      fd.append('target_text', targetText.trim())
    }
    onRun(fd)
  }

  const canRun = domain === 'image'
    ? (inputFile && targetFile && selectedAttacks.length > 0)
    : (audioFile && selectedAttacks.length > 0 && (!audioTargetRequired || targetText.trim()))

  const asrOptions = useMemo(() =>
    (asrModels || []).map(m => typeof m === 'string' ? ({ label: m, value: m }) : m),
    [asrModels]
  )

  return (
    <div className="space-y-4">
      {/* Domain toggle */}
      <div className="flex gap-2 max-w-xs">
        {[['image', 'Image Attacks'], ['audio', 'Audio Attacks']].map(([key, label]) => (
          <button key={key} onClick={() => setDomain(key)}
            className={`flex-1 py-2 rounded-lg text-xs font-semibold transition border
              ${domain === key
                ? 'bg-[var(--accent)]/15 text-[var(--accent)] border-[var(--accent)]/30'
                : 'bg-slate-800/40 text-slate-400 border-slate-600/30 hover:bg-slate-800/60'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Row 1: Input + Attacks side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Input</SectionLabel>
          {domain === 'image' ? (
            <>
              <div className="grid grid-cols-2 gap-4 mb-3">
                <ImgUpload label="Input image" preview={inputPreview} onChange={pickFile(setInputFile, setInputPreview)} />
                <ImgUpload label="Target image" preview={targetPreview} onChange={pickFile(setTargetFile, setTargetPreview)} />
              </div>
              <ModelPicker label="Source model" value={sourceModel} onChange={setSourceModel} models={allModels} />
            </>
          ) : (
            <>
              <div className="mb-3">
                <span className="block text-xs text-slate-400 mb-1">Audio file</span>
                <label className="flex items-center gap-2 px-3 py-2 rounded-lg border-2 border-dashed border-slate-600 hover:border-[var(--accent)]/50 cursor-pointer bg-slate-800/60 transition">
                  <span className="text-xs text-slate-400">{audioName || 'Click to upload .wav / .mp3'}</span>
                  <input type="file" accept="audio/*" onChange={pickAudio} className="hidden" />
                </label>
              </div>
              <div className="mb-3">
                <span className="block text-xs text-slate-400 mb-1">
                  {audioTargetRequired ? 'Target transcription' : 'Reference text'}
                  {!audioTargetRequired && <span className="text-slate-600 ml-1">(optional)</span>}
                </span>
                <input type="text" value={targetText} onChange={e => setTargetText(e.target.value)}
                  placeholder={audioTargetRequired ? 'e.g. open the door' : 'Optional for Speech Jamming'}
                  className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:border-[var(--accent)] outline-none" />
                {!audioTargetRequired && (
                  <p className="mt-1 text-[10px] text-slate-500">
                    Speech Jamming is untargeted. Benchmark success is measured by changing the transcription away from the original.
                  </p>
                )}
              </div>
              <Select label="Source ASR model" value={sourceModel} onChange={setSourceModel} options={asrOptions} className="" />
            </>
          )}
        </Card>

        <Card>
          <SectionLabel>Attacks to Test</SectionLabel>
          <AttackChecklist
            groups={domain === 'image' ? (imageAttackGroups || {}) : (audioAttacks || [])}
            selected={selectedAttacks}
            onChange={setSelectedAttacks}
          />
          {domain === 'image' && selectedAttacks.includes(SARABCRAFT_R1_NAME) && (
            <p className="mt-3 text-[11px] text-slate-500">
              Benchmark runs SarabCraft R1 in standard mode. Use Image Attack or Robustness when you want multi-image transfer tuning.
            </p>
          )}
        </Card>
      </div>

      {/* Row 2: Parameters + Transfer Targets side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Parameters</SectionLabel>
          <ParamModePanel
            domain={domain}
            mode={paramMode} onModeChange={setParamMode}
            preset={preset} onPresetChange={setPreset}
            sweep={sweep} onSweepChange={setSweep}
          />
        </Card>

        <Card>
          <SectionLabel>Transfer Targets</SectionLabel>
          <TransferTargetPicker
            domain={domain}
            targets={transferTargets}
            onChange={setTransferTargets}
            imageLocalTargets={imageLocalTargets}
            imageRemoteTargets={imageRemoteTargets}
            audioRemoteTargets={audioRemoteTargets}
          />
        </Card>
      </div>

      {/* Sticky Run Button Bar — hidden while running */}
      {!loading && (
        <div className="sticky bottom-0 z-20 -mx-1 px-1 pt-3 pb-1 bg-gradient-to-t from-slate-900 via-slate-900/95 to-transparent">
          <div className="flex items-center gap-3 bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 rounded-xl px-4 py-3">
            <RunButton onClick={handleRun} loading={loading} disabled={!canRun && !loading}
              label={`Run (${totalCombos} combo${totalCombos !== 1 ? 's' : ''})`}
              loadingLabel="Running..." />
            {!canRun && (
              <span className="text-[10px] text-slate-500">
                {domain === 'image'
                  ? 'Upload both images + select attacks'
                  : (audioTargetRequired ? 'Upload audio + enter target + select attacks' : 'Upload audio + select attacks')}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
