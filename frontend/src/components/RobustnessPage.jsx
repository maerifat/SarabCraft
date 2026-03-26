import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import { getModels, getAttacks, getAsrModels, runRobustnessSSE, runAudioRobustnessSSE } from '../api/client'
import { Card, SectionLabel, RunButton, ErrorMsg } from './ui/Section'
import { FALLBACK_REGISTRY } from './image/attackRegistry'
import { THREAT_COLORS, THREAT_LABEL } from '../constants/threat'
import { AUDIO_ATTACK_REGISTRY } from './audio/audioAttackRegistry'
import AttackInfoModal from './image/AttackInfoModal'
import ParamRenderer from './image/ParamRenderer'
import { buildAttackParamPayload } from './image/sarabcraftR1'

const AUDIO_ATK_LIST = Object.values(AUDIO_ATTACK_REGISTRY).filter(a => a.robustnessSupported)

function normalizeModelItems(response) {
  if (Array.isArray(response?.items)) {
    return response.items.map(item => ({
      ...item,
      label: item.label || item.display_name || item.value || item.id,
      value: item.value || item.id || item.model_ref,
      group: item.group || item.family || item.task || 'Other',
    })).filter(item => item.value)
  }
  return (response?.models || []).map(value => ({ label: value, value, group: 'Other' }))
}

export default function RobustnessPage() {
  const [domain, setDomain] = useState('image')

  // Image state
  const [inputFile, setInputFile] = useState(null)
  const [inputPreview, setInputPreview] = useState(null)
  const [targetFile, setTargetFile] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [attack, setAttack] = useState('PGD')
  const [modelsList, setModelsList] = useState([])
  const [registry, setRegistry] = useState(FALLBACK_REGISTRY)
  const [selectedModels, setSelectedModels] = useState([])
  const [paramValues, setParamValues] = useState({})

  // Audio state
  const [audioFile, setAudioFile] = useState(null)
  const [audioName, setAudioName] = useState('')
  const [targetText, setTargetText] = useState('')
  const [audioAttack, setAudioAttack] = useState('transcription')
  const [asrModelsList, setAsrModelsList] = useState([])
  const [selectedAsrModels, setSelectedAsrModels] = useState([])
  const [audioParams, setAudioParams] = useState({ epsilon: 0.05, iterations: 300, lr: 0.005 })

  // Shared state
  const [error, setError] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [infoOpen, setInfoOpen] = useState(null)
  const [running, setRunning] = useState(false)
  const [liveResults, setLiveResults] = useState([])
  const [currentModel, setCurrentModel] = useState(null)
  const [currentJobId, setCurrentJobId] = useState('')
  const [progress, setProgress] = useState({ index: 0, total: 0 })
  const [summary, setSummary] = useState(null)
  const abortRef = useRef(null)

  useEffect(() => {
    Promise.all([getModels(), getAttacks(), getAsrModels()]).then(([m, a, asr]) => {
      const imageItems = normalizeModelItems(m)
      setModelsList(imageItems)
      if (a.registry && Object.keys(a.registry).length > 0) setRegistry(a.registry)
      setSelectedModels(imageItems.slice(0, 8).map(item => item.value))
      const asrItems = normalizeModelItems(asr)
      setAsrModelsList(asrItems)
      setSelectedAsrModels(asrItems.slice(0, 2).map(item => item.value))
    }).catch(() => {})
  }, [])

  const attackParams = useMemo(() => registry[attack]?.params || {}, [registry, attack])
  const attackMeta = registry[attack] || {}

  const grouped = useMemo(() => {
    const g = {}
    const lc = searchTerm.toLowerCase()
    Object.entries(registry).forEach(([name, meta]) => {
      if (lc && !name.toLowerCase().includes(lc) && !meta.cat?.toLowerCase().includes(lc) && !meta.authors?.toLowerCase().includes(lc) && !meta.desc?.toLowerCase().includes(lc))
        return
      const cat = meta.cat || 'Other'
      if (!g[cat]) g[cat] = []
      g[cat].push({ name, ...meta })
    })
    return g
  }, [registry, searchTerm])

  const getParam = useCallback((key) => {
    if (key in paramValues) return paramValues[key]
    return attackParams[key]?.default ?? 0
  }, [paramValues, attackParams])

  const setParam = useCallback((key, val) => {
    setParamValues(prev => ({ ...prev, [key]: val }))
  }, [])

  const buildImageAttackParams = useCallback(() => buildAttackParamPayload(attack, attackParams, getParam), [attack, attackParams, getParam])

  const handleAttackChange = useCallback((newAttack) => {
    setAttack(newAttack)
    setParamValues({})
    setInfoOpen(null)
  }, [])

  const pickFile = (setter, previewSetter) => e => {
    const f = e.target.files?.[0]
    setter(f)
    if (f) { const r = new FileReader(); r.onload = ev => previewSetter(ev.target.result); r.readAsDataURL(f) }
  }

  const toggleModel = (val) => {
    setSelectedModels(prev => prev.includes(val) ? prev.filter(m => m !== val) : [...prev, val])
  }

  const handleRun = () => {
    if (!inputFile || !targetFile) { setError('Upload both input and target images'); return }
    if (!selectedModels.length) { setError('Select at least one model'); return }

    setRunning(true)
    setError('')
    setLiveResults([])
    setCurrentModel(null)
    setCurrentJobId('')
    setProgress({ index: 0, total: selectedModels.length })
    setSummary(null)

    const fd = new FormData()
    fd.append('input_file', inputFile)
    fd.append('target_file', targetFile)
    fd.append('attack', attack)
    fd.append('models_json', JSON.stringify(selectedModels))
    const params = buildImageAttackParams()
    fd.append('attack_params_json', JSON.stringify(params))
    fd.append('epsilon', String(params.epsilon ?? 16))
    fd.append('iterations', String(params.iterations ?? 40))

    const handle = runRobustnessSSE(fd, {
      onCreated: (job) => {
        setCurrentJobId(job.job_id)
      },
      onInit: (data) => {
        setProgress({ index: 0, total: data.total_models })
      },
      onProgress: (data) => {
        setCurrentModel(data.model)
        setProgress({ index: data.index, total: data.total })
      },
      onResult: (data) => {
        setLiveResults(prev => [...prev, data])
        setProgress(p => ({ ...p, index: data.index + 1 }))
        setCurrentModel(null)
      },
      onSummary: (data) => {
        setSummary(data)
      },
      onDone: () => {
        setRunning(false)
        setCurrentModel(null)
      },
      onError: (err) => {
        setError(err.message === 'Job cancelled' ? 'Job cancelled. Open Jobs to resume later.' : (err.message || 'Connection lost'))
        setRunning(false)
      },
    })
    abortRef.current = handle
  }

  const handleAbort = () => {
    abortRef.current?.abort()
    setRunning(false)
    setCurrentModel(null)
    setError('Cancellation requested. Open Jobs to monitor or resume later.')
  }

  const activeAudioAttack = AUDIO_ATTACK_REGISTRY[audioAttack] || {}
  const audioTargetRequired = !!activeAudioAttack.requiresTargetText

  const canRun = domain === 'image'
    ? (inputFile && targetFile && selectedModels.length > 0)
    : (audioFile && selectedAsrModels.length > 0 && (!audioTargetRequired || targetText.trim()))

  const handleRunAudio = () => {
    if (!audioFile) { setError('Upload audio to test'); return }
    if (audioTargetRequired && !targetText.trim()) { setError('Enter target text for this attack'); return }
    if (!selectedAsrModels.length) { setError('Select at least one ASR model'); return }

    setRunning(true); setError(''); setLiveResults([]); setCurrentModel(null); setCurrentJobId('')
    setProgress({ index: 0, total: selectedAsrModels.length }); setSummary(null)

    const atkMeta = AUDIO_ATTACK_REGISTRY[audioAttack]
    const fd = new FormData()
    fd.append('audio_file', audioFile)
    fd.append('target_text', audioTargetRequired ? targetText.trim() : '')
    fd.append('attack', atkMeta?.name || 'Targeted Transcription')
    fd.append('models_json', JSON.stringify(selectedAsrModels))
    fd.append('attack_params_json', JSON.stringify(audioParams))

    const handle = runAudioRobustnessSSE(fd, {
      onCreated: (job) => setCurrentJobId(job.job_id),
      onInit: (data) => setProgress({ index: 0, total: data.total_models }),
      onProgress: (data) => { setCurrentModel(data.model); setProgress({ index: data.index, total: data.total }) },
      onResult: (data) => { setLiveResults(prev => [...prev, data]); setProgress(p => ({ ...p, index: data.index + 1 })); setCurrentModel(null) },
      onSummary: (data) => setSummary(data),
      onDone: () => { setRunning(false); setCurrentModel(null) },
      onError: (err) => { setError(err.message === 'Job cancelled' ? 'Job cancelled. Open Jobs to resume later.' : (err.message || 'Connection lost')); setRunning(false) },
    })
    abortRef.current = handle
  }

  const modelGroups = useMemo(() => {
    const groups = {}
    modelsList.forEach(item => {
      const cat = item.group || 'Other'
      ;(groups[cat] = groups[cat] || []).push(item)
    })
    return groups
  }, [modelsList])

  const pct = progress.total > 0 ? Math.round((progress.index / progress.total) * 100) : 0
  const hasResults = liveResults.length > 0
  const successes = liveResults.filter(r => r.success).length

  const audioGrouped = useMemo(() => {
    const g = {}
    const lc = searchTerm.toLowerCase()
    AUDIO_ATK_LIST.forEach(atk => {
      if (lc && !atk.name.toLowerCase().includes(lc) && !atk.cat?.toLowerCase().includes(lc) && !atk.desc?.toLowerCase().includes(lc)) return
      const cat = atk.cat || 'Other'
      if (!g[cat]) g[cat] = []
      g[cat].push(atk)
    })
    return g
  }, [searchTerm])

  const pickAudio = e => { const f = e.target.files?.[0]; setAudioFile(f); setAudioName(f?.name || '') }

  const activeModels = domain === 'image' ? selectedModels : selectedAsrModels

  return (
    <div className="space-y-4">
      {/* Domain Toggle */}
      <div className="flex gap-1 p-0.5 bg-slate-800/60 rounded-lg border border-slate-700/50 w-fit">
        {[['image', 'Image Models'], ['audio', 'Audio Models']].map(([key, label]) => (
          <button key={key} onClick={() => { if (!running) { setDomain(key); setLiveResults([]); setSummary(null); setError('') } }}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all ${domain === key
              ? 'bg-[var(--accent)]/15 text-[var(--accent)] border border-[var(--accent)]/30'
              : 'text-slate-400 hover:text-slate-200 border border-transparent'}`}>{label}</button>
        ))}
      </div>
      {/* ====== IMAGE DOMAIN ====== */}
      {domain === 'image' && (<>
      {/* Row 1: Input + Attack Method */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Input</SectionLabel>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <ImgUpload label="Input image" preview={inputPreview} onChange={pickFile(setInputFile, setInputPreview)} />
            <ImgUpload label="Target image" preview={targetPreview} onChange={pickFile(setTargetFile, setTargetPreview)} />
          </div>
        </Card>

        <Card>
          <SectionLabel>Attack Method</SectionLabel>
          <AttackSelector grouped={grouped} attack={attack} searchTerm={searchTerm} setSearchTerm={setSearchTerm}
            onSelect={handleAttackChange} onInfo={name => setInfoOpen(infoOpen === name ? null : name)} />
        </Card>
      </div>

      <AttackInfoModal name={infoOpen} meta={infoOpen ? registry[infoOpen] : null} onClose={() => setInfoOpen(null)} />

      {/* Row 2: Parameters + Models */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        {Object.keys(attackParams).length > 0 ? (
          <Card>
            <SectionLabel>
              Parameters — {attack}
              <span className={`ml-2 text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[attackMeta.threat] || THREAT_COLORS.whitebox}`}>
                {THREAT_LABEL[attackMeta.threat] || 'White-box'}
              </span>
            </SectionLabel>
            <ParamRenderer attackName={attack} params={attackParams} getParam={getParam} setParam={setParam} />
          </Card>
        ) : (
          <div />
        )}

        <Card>
          <SectionLabel>Models to Test ({selectedModels.length} selected)</SectionLabel>
          <div className="flex flex-wrap gap-1.5 mb-3">
            {Object.keys(modelGroups).map(cat => {
              const vals = modelGroups[cat].map(m => m.value)
              const selCount = vals.filter(v => selectedModels.includes(v)).length
              const allSel = vals.length > 0 && selCount === vals.length
              const someSel = selCount > 0 && !allSel
              return (
                <button key={cat} onClick={() => {
                  setSelectedModels(prev => allSel
                    ? prev.filter(m => !vals.includes(m))
                    : [...new Set([...prev, ...vals])]
                  )
                }}
                  className={`text-[10px] px-2 py-0.5 rounded border transition ${allSel
                    ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20 hover:bg-[var(--accent)]/20'
                    : someSel
                      ? 'bg-[var(--accent)]/5 text-[var(--accent)]/70 border-[var(--accent)]/10 hover:bg-[var(--accent)]/15'
                      : 'bg-slate-700/40 text-slate-400 border-slate-600/30 hover:text-[var(--accent)] hover:bg-slate-700/60'}`}>
                  {cat}
                  {someSel && <span className="ml-1 text-[9px] opacity-60">{selCount}/{vals.length}</span>}
                </button>
              )
            })}
            <button onClick={() => {
              const all = modelsList.map(item => item.value)
              setSelectedModels(selectedModels.length === all.length ? [] : all)
            }}
              className={`text-[10px] px-2 py-0.5 rounded border transition ${selectedModels.length === modelsList.length && modelsList.length > 0
                ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20 hover:bg-[var(--accent)]/20'
                : 'bg-slate-700/40 text-slate-400 border-slate-600/30 hover:text-[var(--accent)] hover:bg-slate-700/60'}`}>All</button>
            <button onClick={() => setSelectedModels([])}
              className={`text-[10px] px-2 py-0.5 rounded border transition ${selectedModels.length === 0
                ? 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20 hover:bg-[var(--accent)]/20'
                : 'bg-slate-700/40 text-slate-400 border-slate-600/30 hover:text-[var(--accent)] hover:bg-slate-700/60'}`}>Clear</button>
          </div>
          <div className="max-h-64 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/30 divide-y divide-slate-800/40">
            {modelsList.map(item => (
              <label key={item.value} className={`flex items-center gap-2.5 px-3 py-2 cursor-pointer text-xs transition-colors
                ${selectedModels.includes(item.value)
                  ? 'bg-[var(--accent)]/8 text-[var(--accent)]'
                  : 'text-slate-400 hover:bg-slate-800/40'}`}>
                <input type="checkbox" checked={selectedModels.includes(item.value)} onChange={() => toggleModel(item.value)}
                  className="w-3.5 h-3.5 rounded accent-[var(--accent)] shrink-0" />
                <span className="truncate">{item.label.replace(/^\[[^\]]+\]\s*/, '')}</span>
              </label>
            ))}
          </div>
        </Card>
      </div>
      </>)}

      {/* ====== AUDIO DOMAIN ====== */}
      {domain === 'audio' && (<>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Audio Input</SectionLabel>
          <div className="space-y-3">
            <label className="flex items-center gap-3 p-3 border-2 border-dashed border-slate-600 hover:border-[var(--accent)]/50 rounded-lg cursor-pointer bg-slate-800/60 transition">
              <svg className="w-5 h-5 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
              <span className="text-xs text-slate-400 truncate">{audioName || 'Click to upload .wav / .mp3'}</span>
              <input type="file" accept="audio/*" onChange={pickAudio} className="hidden" />
            </label>
            <div>
              <span className="block text-xs text-slate-400 mb-1">
                {audioTargetRequired ? 'Target transcription' : 'Reference text'}
                {!audioTargetRequired && <span className="text-slate-600 ml-1">(optional)</span>}
              </span>
              <input type="text" value={targetText} onChange={e => setTargetText(e.target.value)}
                placeholder={audioTargetRequired ? 'e.g. open the door' : 'Optional text for comparison'}
                className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] outline-none" />
              {!audioTargetRequired && (
                <p className="mt-1 text-[10px] text-slate-500">
                  This attack is untargeted. Success means the transcription changes away from the original.
                </p>
              )}
            </div>
          </div>
        </Card>

        <Card>
          <SectionLabel>Audio Attack Method</SectionLabel>
          <AudioAttackSelector grouped={audioGrouped} selected={audioAttack} searchTerm={searchTerm}
            setSearchTerm={setSearchTerm} onSelect={id => { setAudioAttack(id); setSearchTerm('') }} />
        </Card>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Audio Attack Parameters</SectionLabel>
          <div className="space-y-3">
            <ParamSlider label="Epsilon" value={audioParams.epsilon} min={0.01} max={0.3} step={0.01}
              onChange={v => setAudioParams(p => ({ ...p, epsilon: v }))} />
            <ParamSlider label="Iterations" value={audioParams.iterations} min={50} max={1000} step={50}
              onChange={v => setAudioParams(p => ({ ...p, iterations: v }))} />
            <ParamSlider label="Learning Rate" value={audioParams.lr} min={0.001} max={0.05} step={0.001}
              onChange={v => setAudioParams(p => ({ ...p, lr: v }))} />
          </div>
        </Card>

        <Card>
          <SectionLabel>ASR Models to Test ({selectedAsrModels.length} selected)</SectionLabel>
          <div className="max-h-64 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/30 divide-y divide-slate-800/40">
            {asrModelsList.map(item => {
              const sel = selectedAsrModels.includes(item.value)
              return (
                <label key={item.value} className={`flex items-center gap-2.5 px-3 py-2.5 cursor-pointer text-xs transition-colors
                  ${sel ? 'bg-[var(--accent)]/8 text-[var(--accent)]' : 'text-slate-400 hover:bg-slate-800/40'}`}>
                  <input type="checkbox" checked={sel} onChange={() => setSelectedAsrModels(prev => sel ? prev.filter(m => m !== item.value) : [...prev, item.value])}
                    className="w-3.5 h-3.5 rounded accent-[var(--accent)] shrink-0" />
                  <span className="truncate">{item.label.replace(/^\[[^\]]+\]\s*/, '')}</span>
                  <span className="text-[10px] text-slate-600 ml-auto shrink-0">{item.group || ''}</span>
                </label>
              )
            })}
          </div>
        </Card>
      </div>
      </>)}

      {/* Sticky Run / Abort Bar */}
      {!running && (
        <div className="sticky bottom-0 z-20 -mx-1 px-1 pt-3 pb-1 bg-gradient-to-t from-slate-900 via-slate-900/95 to-transparent">
          <div className="flex items-center gap-3 bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 rounded-xl px-4 py-3">
            <RunButton onClick={domain === 'image' ? handleRun : handleRunAudio} disabled={!canRun}
              label={`Test ${activeModels.length} Model${activeModels.length !== 1 ? 's' : ''}`}
              loadingLabel="Testing..." />
            {!canRun && (
              <span className="text-[10px] text-slate-500">
                {domain === 'image'
                  ? 'Upload both images + select models'
                  : (audioTargetRequired ? 'Upload audio + enter target text + select models' : 'Upload audio + select models')}
              </span>
            )}
            <ErrorMsg msg={error} />
          </div>
        </div>
      )}
      {running && (
        <div className="sticky bottom-0 z-20 -mx-1 px-1 pt-3 pb-1 bg-gradient-to-t from-slate-900 via-slate-900/95 to-transparent">
          <div className="flex items-center gap-3 bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 rounded-xl px-4 py-3">
            <button onClick={handleAbort}
              className="px-8 py-3 rounded-lg text-sm font-bold bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25 transition">
              Abort
            </button>
            <span className="text-xs text-slate-400">
              Testing {progress.index}/{progress.total}
              {currentModel && <> — <span className="text-slate-300">{currentModel}</span></>}
            </span>
          </div>
        </div>
      )}

      {currentJobId && (
        <Card className="border-[var(--accent)]/20">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <SectionLabel>Durable Job</SectionLabel>
              <p className="text-sm font-semibold text-slate-200">Robustness run is tracked as a resumable background job.</p>
              <p className="text-xs text-slate-500 mt-1">Job ID: {currentJobId}</p>
            </div>
            <Link
              to={`/jobs?job=${currentJobId}`}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 hover:bg-cyan-500/20 transition"
            >
              Open in Jobs
            </Link>
          </div>
          <p className="mt-3 text-[11px] text-slate-500">
            Use Jobs for persistent status, cancellation, and resume after refresh or navigation.
          </p>
        </Card>
      )}

      {/* Live Progress */}
      {(running || hasResults) && (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              {running && (
                <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--accent)' }} />
              )}
              <span className="text-sm font-semibold text-slate-200">
                {running ? 'Testing in progress...' : 'Results'}
              </span>
            </div>
            <div className="flex items-center gap-4 text-[10px] uppercase tracking-wider text-slate-500">
              <span>{progress.index} / {progress.total} done</span>
              {hasResults && (
                <>
                  <span className="text-red-400">{successes} vulnerable</span>
                  <span className="text-emerald-400">{progress.index - successes} robust</span>
                </>
              )}
            </div>
          </div>

          <div className="relative h-1.5 bg-slate-700/50 rounded-full overflow-hidden mb-1">
            <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
              style={{ width: `${pct}%`, background: 'var(--accent)', opacity: 0.8 }} />
          </div>
          {currentModel && (
            <p className="text-[10px] text-slate-500 mb-4">
              Running on <span className="text-slate-300">{currentModel}</span>...
            </p>
          )}
          {!currentModel && !running && <div className="mb-4" />}

          {hasResults && (
            <div className="space-y-1">
              {liveResults.map((r, i) => {
                const name = r.model?.split('/').pop() || r.model_key?.replace(/^\[.*?\]\s*/, '') || r.model
                const isAudio = domain === 'audio'
                return (
                  <div key={r.model || r.model_key || i} className="flex items-center gap-3 animate-in"
                    style={{ animation: 'fadeSlideIn 0.3s ease-out' }}>
                    <span className="text-[10px] text-slate-400 w-40 truncate shrink-0 font-mono" title={r.model}>{name}</span>
                    <div className="flex-1 bg-slate-700/30 rounded-full h-5 overflow-hidden">
                      {r.error ? (
                        <div className="h-full flex items-center px-3 text-[10px] text-red-400">Error: {r.error.slice(0, 50)}</div>
                      ) : isAudio ? (
                        <div className={`h-full rounded-full flex items-center px-3 ${r.success ? 'bg-red-500/40' : 'bg-emerald-500/30'}`}
                          style={{ width: '100%', transition: 'width 0.5s ease-out' }}>
                          <span className="text-[10px] font-medium text-white whitespace-nowrap truncate">
                            {r.success ? `→ "${r.result_text}"` : `"${r.result_text || r.original_text}"`}
                          </span>
                        </div>
                      ) : (
                        <div className={`h-full rounded-full flex items-center px-3 ${r.success ? 'bg-red-500/40' : 'bg-emerald-500/30'}`}
                          style={{ width: `${Math.max((r.top_adv_conf || 0) * 100, 12)}%`, transition: 'width 0.5s ease-out' }}>
                          <span className="text-[10px] font-medium text-white whitespace-nowrap">
                            {r.success ? `→ ${r.adversarial_class}` : r.adversarial_class}
                          </span>
                        </div>
                      )}
                    </div>
                    {isAudio && <span className="text-[10px] text-slate-500 w-12 text-right shrink-0 font-mono">{r.snr_db ? `${r.snr_db}dB` : '—'}</span>}
                    <span className={`text-[10px] w-10 text-right shrink-0 font-mono tabular-nums ${r.success ? 'text-red-400' : r.error ? 'text-slate-600' : 'text-emerald-400'}`}>
                      {r.elapsed_ms ? `${(r.elapsed_ms / 1000).toFixed(1)}s` : '—'}
                    </span>
                    <span className={`w-5 text-center text-xs ${r.success ? 'text-red-400' : r.error ? 'text-slate-600' : 'text-emerald-400'}`}>
                      {r.error ? '✕' : r.success ? '✗' : '✓'}
                    </span>
                  </div>
                )
              })}
              {running && currentModel && (
                <div className="flex items-center gap-3 opacity-50">
                  <span className="text-[10px] text-slate-500 w-40 truncate shrink-0 font-mono">{currentModel}</span>
                  <div className="flex-1 bg-slate-700/30 rounded-full h-5 overflow-hidden">
                    <div className="h-full rounded-full animate-pulse" style={{ width: '30%', background: 'var(--accent)', opacity: 0.15 }} />
                  </div>
                  <span className="text-[10px] w-10 text-right text-slate-600 font-mono">...</span>
                  <span className="w-5 text-center text-xs text-slate-600">⋯</span>
                </div>
              )}
            </div>
          )}

          {summary && !running && (
            <div className="mt-6 pt-4 border-t border-slate-700/50">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '0.75rem' }} className="mb-5">
                <StatCard value={summary.total_models} label="Models Tested" color="text-white" />
                <StatCard value={summary.successful_transfers ?? summary.successful_attacks ?? 0} label={domain === 'audio' ? 'Attack Succeeded' : 'Vulnerable'} color="text-red-400" />
                <StatCard value={`${((summary.transfer_rate ?? summary.attack_success_rate ?? 0) * 100).toFixed(1)}%`}
                  label={domain === 'audio' ? 'Success Rate' : 'Transfer Rate'}
                  color={(summary.transfer_rate ?? summary.attack_success_rate ?? 0) > 0.5 ? 'text-red-400' : 'text-emerald-400'} />
              </div>

              <details className="group">
                <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-300 select-none">Detailed metrics table</summary>
                <div className="mt-3 overflow-x-auto">
                  {domain === 'audio' ? (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-500 border-b border-slate-700/50 text-[10px] uppercase tracking-wider">
                        <th className="text-left py-2 pr-3">Model</th>
                        <th className="text-left py-2 pr-3">Result</th>
                        <th className="text-left py-2 pr-3">Original Text</th>
                        <th className="text-left py-2 pr-3">Output Text</th>
                        <th className="text-right py-2 pr-3">SNR (dB)</th>
                        <th className="text-right py-2">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(summary.results || []).map(r => (
                        <tr key={r.model || r.model_key} className="border-b border-slate-800/50">
                          <td className="py-1.5 pr-3 text-slate-400 truncate max-w-[140px]">{(r.model_key || r.model || '').replace(/^\[.*?\]\s*/, '')}</td>
                          <td className="py-1.5 pr-3">
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium
                              ${r.success ? 'bg-red-500/15 text-red-400' : r.error ? 'bg-slate-600/20 text-slate-500' : 'bg-emerald-500/15 text-emerald-400'}`}>
                              {r.error ? 'ERROR' : r.success ? (r.evaluation_mode === 'untargeted' ? 'DISRUPTED' : 'FOOLED') : (r.evaluation_mode === 'untargeted' ? 'UNCHANGED' : 'ROBUST')}
                            </span>
                          </td>
                          <td className="py-1.5 pr-3 text-slate-500 truncate max-w-[150px]">{r.original_text || '—'}</td>
                          <td className="py-1.5 pr-3 text-slate-300 truncate max-w-[150px]">{r.result_text || '—'}</td>
                          <td className="py-1.5 pr-3 text-right text-slate-500">{r.snr_db ?? '—'}</td>
                          <td className="py-1.5 text-right text-slate-500">{r.elapsed_ms ? `${(r.elapsed_ms / 1000).toFixed(1)}s` : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  ) : (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-500 border-b border-slate-700/50 text-[10px] uppercase tracking-wider">
                        <th className="text-left py-2 pr-3">Model</th>
                        <th className="text-left py-2 pr-3">Result</th>
                        <th className="text-left py-2 pr-3">Original</th>
                        <th className="text-left py-2 pr-3">Adversarial</th>
                        <th className="text-right py-2 pr-3">L2</th>
                        <th className="text-right py-2 pr-3">SSIM</th>
                        <th className="text-right py-2">PSNR</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(summary.results || []).map(r => (
                        <tr key={r.model} className="border-b border-slate-800/50">
                          <td className="py-1.5 pr-3 text-slate-400 truncate max-w-[140px]">{r.model?.split('/').pop()}</td>
                          <td className="py-1.5 pr-3">
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium
                              ${r.success ? 'bg-red-500/15 text-red-400' : r.error ? 'bg-slate-600/20 text-slate-500' : 'bg-emerald-500/15 text-emerald-400'}`}>
                              {r.error ? 'ERROR' : r.success ? 'VULNERABLE' : 'ROBUST'}
                            </span>
                          </td>
                          <td className="py-1.5 pr-3 text-slate-500 truncate max-w-[100px]">{r.original_class || '—'}</td>
                          <td className="py-1.5 pr-3 text-slate-300 truncate max-w-[100px]">{r.adversarial_class || '—'}</td>
                          <td className="py-1.5 pr-3 text-right text-slate-500">{r.metrics?.l2?.toFixed(4) ?? '—'}</td>
                          <td className="py-1.5 pr-3 text-right text-slate-500">{r.metrics?.ssim?.toFixed(3) ?? '—'}</td>
                          <td className="py-1.5 text-right text-slate-500">{r.metrics?.psnr?.toFixed(1) ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  )}
                </div>
              </details>
            </div>
          )}
        </Card>
      )}

      <style>{`
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}


function AttackSelector({ grouped, attack, searchTerm, setSearchTerm, onSelect, onInfo }) {
  return (
    <div className="space-y-2">
      <div className="relative">
        <input type="text" placeholder="Search attacks..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] focus:outline-none" />
        {searchTerm && <button onClick={() => setSearchTerm('')} className="absolute right-2 top-2 text-slate-500 hover:text-slate-300 text-sm">✕</button>}
      </div>
      <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/40 divide-y divide-slate-800/50">
        {Object.entries(grouped).map(([cat, attacks]) => (
          <div key={cat}>
            <div className="px-3 py-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-wider bg-slate-800/40 sticky top-0 z-10">{cat}</div>
            {attacks.map(a => {
              const sel = a.name === attack
              return (
                <div key={a.name} onClick={() => onSelect(a.name)}
                  className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${sel ? 'bg-[var(--accent)]/10 border-l-2 border-[var(--accent)]' : 'hover:bg-slate-800/50 border-l-2 border-transparent'}`}>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm truncate ${sel ? 'text-[var(--accent)] font-semibold' : 'text-slate-200'}`}>{a.name}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium shrink-0 ${THREAT_COLORS[a.threat] || THREAT_COLORS.whitebox}`}>{THREAT_LABEL[a.threat] || 'White-box'}</span>
                      <span className="text-[10px] text-slate-600 shrink-0">{a.year}</span>
                    </div>
                    <div className="text-[10px] text-slate-500 truncate">{a.authors} — {a.norm}</div>
                  </div>
                  <button onClick={e => { e.stopPropagation(); onInfo(a.name) }}
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
  )
}

function StatCard({ value, label, color = 'text-white' }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3 text-center">
      <div className={`text-xl font-black ${color}`}>{value}</div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
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

function AudioAttackSelector({ grouped, selected, searchTerm, setSearchTerm, onSelect }) {
  return (
    <div className="space-y-2">
      <div className="relative">
        <input type="text" placeholder="Search audio attacks..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] focus:outline-none" />
        {searchTerm && <button onClick={() => setSearchTerm('')} className="absolute right-2 top-2 text-slate-500 hover:text-slate-300 text-sm">✕</button>}
      </div>
      <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/40 divide-y divide-slate-800/50">
        {Object.entries(grouped).map(([cat, attacks]) => (
          <div key={cat}>
            <div className="px-3 py-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-wider bg-slate-800/40 sticky top-0 z-10">{cat}</div>
            {attacks.map(a => {
              const sel = a.id === selected
              return (
                <div key={a.id} onClick={() => onSelect(a.id)}
                  className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${sel ? 'bg-[var(--accent)]/10 border-l-2 border-[var(--accent)]' : 'hover:bg-slate-800/50 border-l-2 border-transparent'}`}>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm truncate ${sel ? 'text-[var(--accent)] font-semibold' : 'text-slate-200'}`}>{a.name}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium shrink-0 ${THREAT_COLORS[a.threat] || THREAT_COLORS.whitebox}`}>{THREAT_LABEL[a.threat] || 'White-box'}</span>
                      <span className="text-[10px] text-slate-600 shrink-0">{a.year}</span>
                    </div>
                    <div className="text-[10px] text-slate-500 truncate">{a.authors} — {a.norm}</div>
                  </div>
                </div>
              )
            })}
          </div>
        ))}
        {Object.keys(grouped).length === 0 && <div className="px-3 py-4 text-xs text-slate-500 text-center">No attacks match "{searchTerm}"</div>}
      </div>
    </div>
  )
}

function ParamSlider({ label, value, min, max, step, onChange }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400">{label}</span>
        <input type="number" value={value} min={min} max={max} step={step}
          onChange={e => onChange(parseFloat(e.target.value) || min)}
          className="w-20 bg-slate-700/80 border border-slate-600 rounded px-2 py-0.5 text-xs text-slate-200 text-right focus:border-[var(--accent)] outline-none" />
      </div>
      <input type="range" value={value} min={min} max={max} step={step}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1 accent-[var(--accent)] cursor-pointer" />
    </div>
  )
}
