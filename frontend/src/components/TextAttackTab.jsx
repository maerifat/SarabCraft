import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { getTextModels, getTextMethods, runTextAttack, createAbortable, cancelJobById } from '../api/client'
import { Card, SectionLabel, Select, RunButton, ErrorMsg } from './ui/Section'
import AttackInfoModal from './text/AttackInfoModal'
import { useTextAttack } from './TextAttackContext'

import { THREAT_COLORS, THREAT_LABEL } from '../constants/threat'

/* ── Category level colours ────────────────────────────────────── */
const LEVEL_COLORS = {
  'Character-Level': 'text-rose-400',
  'Word-Level':      'text-sky-400',
  'Sentence-Level':  'text-violet-400',
}

export default function TextAttackTab() {
  /* ── Persistent state (survives tab switch) ───── */
  const {
    model, setModel,
    attack, setAttack,
    text, setText,
    targetLabel, setTargetLabel,
    params, setParams,
    loading, setLoading,
    error, setError,
    result, setResult,
    jobId, setJobId,
    abortRef,
  } = useTextAttack()

  /* ── Ephemeral UI state (re-fetched on mount) ── */
  const [models, setModels]             = useState([])
  const [defaultModel, setDefaultModel] = useState('')
  const [registry, setRegistry]         = useState({})
  const [searchTerm, setSearchTerm]     = useState('')
  const [infoOpen, setInfoOpen]         = useState(null)

  /* ── Fetch models + methods on mount ──────────── */
  useEffect(() => {
    Promise.allSettled([getTextModels(), getTextMethods()]).then(([m, a]) => {
      if (m.status === 'fulfilled') {
        const d = m.value
        setModels(d.models || [])
        setDefaultModel(d.default || '')
        setModel(prev => prev || d.default || '')
      }
      if (a.status === 'fulfilled') {
        const attacks = a.value.attacks || {}
        setRegistry(attacks)
        if (!attack && Object.keys(attacks).length) setAttack(Object.keys(attacks)[0])
      }
    })
  }, [])

  /* ── Grouped attacks for selector ─────────────── */
  const grouped = useMemo(() => {
    const g = {}
    const lc = searchTerm.toLowerCase()
    Object.entries(registry).forEach(([name, meta]) => {
      if (lc && !name.toLowerCase().includes(lc) && !meta.category?.toLowerCase().includes(lc) && !meta.description?.toLowerCase().includes(lc))
        return
      const cat = meta.category || 'Other'
      if (!g[cat]) g[cat] = []
      g[cat].push({ name, ...meta })
    })
    return g
  }, [registry, searchTerm])

  const attackMeta = registry[attack] || {}

  /* ── Initialize params when attack changes ────── */
  useEffect(() => {
    if (!attack || !attackMeta.params) return
    const defaults = {}
    Object.entries(attackMeta.params).forEach(([key, spec]) => {
      defaults[key] = spec.default
    })
    setParams(defaults)
  }, [attack, attackMeta.params])

  /* ── Run attack ──────────────────────────────── */
  const handleRun = async () => {
    if (!text.trim()) { setError('Enter text to attack'); return }
    abortRef.current?.abort()
    const { signal, abort } = createAbortable()
    abortRef.current = { abort }

    setLoading(true)
    setError('')
    setResult(null)
    setJobId('')

    try {
      const fd = new FormData()
      fd.append('text', text)
      fd.append('model', model || defaultModel)
      fd.append('attack', attack)
      if (targetLabel.trim()) fd.append('target_label', targetLabel.trim())
      fd.append('params', JSON.stringify(params))

      const res = await runTextAttack(fd, {
        signal,
        cancelOnAbort: false,
        onCreated: job => setJobId(job.job_id),
      })
      setResult(res)
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message)
    } finally {
      setLoading(false)
      abortRef.current = null
    }
  }

  const handleCancel = async () => {
    abortRef.current?.abort()
    abortRef.current = null
    setLoading(false)
    setError('Cancellation requested.')
    if (jobId) { try { await cancelJobById(jobId) } catch {} }
  }

  /* ── Render ──────────────────────────────────── */
  return (
    <div className="space-y-4">
      {/* Row 1: Input + Attack selector */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        {/* Left: Input */}
        <Card>
          <SectionLabel>Input</SectionLabel>
          <div className="space-y-3">
            <div>
              <span className="block text-xs text-slate-400 mb-1">Text to attack</span>
              <textarea
                rows={5}
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder="Enter your text here..."
                className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] focus:outline-none resize-none"
              />
            </div>
            <Select
              label="Model"
              value={model}
              onChange={setModel}
              options={models.length ? models.map(m => ({ value: m.model_id, label: m.display_name })) : [{ value: '', label: 'Loading…' }]}
            />
            <div>
              <span className="block text-xs text-slate-400 mb-1">Target label (optional, for targeted attacks)</span>
              <input
                type="text"
                value={targetLabel}
                onChange={e => setTargetLabel(e.target.value)}
                placeholder="e.g. POSITIVE"
                className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] focus:outline-none"
              />
            </div>
          </div>
        </Card>

        {/* Right: Attack selector */}
        <Card>
          <SectionLabel>Attack Method</SectionLabel>
          <AttackSelector
            grouped={grouped}
            attack={attack}
            searchTerm={searchTerm}
            setSearchTerm={setSearchTerm}
            onSelect={setAttack}
            onInfo={name => setInfoOpen(infoOpen === name ? null : name)}
          />
        </Card>
      </div>

      <AttackInfoModal name={infoOpen} meta={infoOpen ? registry[infoOpen] : null} onClose={() => setInfoOpen(null)} />

      {/* Attack info bar */}
      {attackMeta.description && (
        <div className="flex items-center gap-3 px-1 flex-wrap">
          <span className="text-sm font-semibold text-slate-200">{attack}</span>
          <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[attackMeta.threat_model] || THREAT_COLORS.blackbox}`}>
            {THREAT_LABEL[attackMeta.threat_model] || attackMeta.threat_model}
          </span>
          <span className={`text-[10px] font-medium ${LEVEL_COLORS[attackMeta.category] || 'text-slate-400'}`}>{attackMeta.category}</span>
          {attackMeta.authors && <span className="text-[10px] text-slate-500">{attackMeta.authors} · {attackMeta.year}</span>}
          {attackMeta.arxiv && (
            <a href={`https://arxiv.org/abs/${attackMeta.arxiv}`} target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-[var(--accent)] transition-colors"
              title={attackMeta.paper || 'View paper'}>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>
              Paper
            </a>
          )}
        </div>
      )}

      {/* Parameters */}
      {attackMeta.params && Object.keys(attackMeta.params).length > 0 && (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <SectionLabel>Parameters — {attack}</SectionLabel>
            <button
              onClick={() => {
                const defaults = {}
                Object.entries(attackMeta.params).forEach(([key, spec]) => {
                  defaults[key] = spec.default
                })
                setParams(defaults)
              }}
              className="text-xs text-slate-400 hover:text-[var(--accent)] transition-colors flex items-center gap-1"
              title="Reset all parameters to default values"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
              </svg>
              Reset to Defaults
            </button>
          </div>
          <AttackParameters
            params={attackMeta.params}
            values={params}
            onChange={setParams}
          />
        </Card>
      )}

      {/* Run button */}
      <div className="flex items-center gap-4">
        <RunButton onClick={handleRun} loading={loading} label="Run Text Attack" />
        {loading && (
          <button onClick={handleCancel}
            className="px-4 py-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg text-sm hover:bg-red-500/20 transition font-medium">
            Cancel
          </button>
        )}
        <ErrorMsg msg={error} />
      </div>

      {/* Job status */}
      {(loading || jobId) && (
        <Card className="border-[var(--accent)]/20">
          <SectionLabel>Job Status</SectionLabel>
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <p className="text-sm font-semibold text-slate-200">Running as a background job.</p>
              <p className="text-xs text-slate-500 mt-1">{jobId ? `Job ID: ${jobId}` : 'Submitting…'}</p>
            </div>
            {jobId && (
              <Link to={`/jobs?job=${jobId}`}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 hover:bg-cyan-500/20 transition">
                Open in Jobs
              </Link>
            )}
          </div>
        </Card>
      )}

      {/* ── Results ──────────────────────────────── */}
      {result && <TextAttackResult result={result} attack={attack} model={model} />}
    </div>
  )
}


/* ── Attack parameters component ─────────────────── */
function AttackParameters({ params, values, onChange }) {
  const handleChange = (key, value) => {
    onChange({ ...values, [key]: value })
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(params).map(([key, spec]) => {
        const val = values[key] ?? spec.default
        
        if (spec.type === 'select') {
          return (
            <div key={key}>
              <label className="block text-xs text-slate-400 mb-1.5 font-medium">
                {key.replace(/_/g, ' ')}
              </label>
              <select
                value={val}
                onChange={e => handleChange(key, e.target.value)}
                className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:border-[var(--accent)] focus:outline-none"
              >
                {spec.options?.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
          )
        }

        if (spec.type === 'int' || spec.type === 'float') {
          const step = spec.step || (spec.type === 'int' ? 1 : 0.01)
          return (
            <div key={key}>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs text-slate-400 font-medium">
                  {key.replace(/_/g, ' ')}
                </label>
                <span className="text-xs text-[var(--accent)] font-mono">{val}</span>
              </div>
              <input
                type="range"
                min={spec.min}
                max={spec.max}
                step={step}
                value={val}
                onChange={e => handleChange(key, spec.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-[var(--accent)]"
              />
              <div className="flex justify-between text-[10px] text-slate-600 mt-1">
                <span>{spec.min}</span>
                <span>{spec.max}</span>
              </div>
            </div>
          )
        }

        return null
      })}
    </div>
  )
}


/* ── Attack selector component ───────────────────── */
function AttackSelector({ grouped, attack, searchTerm, setSearchTerm, onSelect, onInfo }) {
  return (
    <div className="space-y-2">
      <label className="text-xs text-slate-400 font-medium">Attack Method</label>
      <div className="relative">
        <input type="text" placeholder="Search attacks..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-[var(--accent)] focus:outline-none" />
        {searchTerm && <button onClick={() => setSearchTerm('')} className="absolute right-2 top-2 text-slate-500 hover:text-slate-300 text-sm">✕</button>}
      </div>
      <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-700/50 bg-slate-900/40 divide-y divide-slate-800/50">
        {Object.entries(grouped).map(([cat, attacks]) => (
          <div key={cat}>
            <div className={`px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider bg-slate-800/40 sticky top-0 z-10 ${LEVEL_COLORS[cat] || 'text-slate-500'}`}>{cat}</div>
            {attacks.map(a => {
              const sel = a.name === attack
              return (
                <div key={a.name} onClick={() => onSelect(a.name)}
                  className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${sel ? 'bg-[var(--accent)]/10 border-l-2 border-[var(--accent)]' : 'hover:bg-slate-800/50 border-l-2 border-transparent'}`}>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm truncate ${sel ? 'text-[var(--accent)] font-semibold' : 'text-slate-200'}`}>{a.name}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[a.threat_model] || THREAT_COLORS.blackbox}`}>
                        {THREAT_LABEL[a.threat_model] || a.threat_model}
                      </span>
                      {a.year && <span className="text-[10px] text-slate-600">{a.year}</span>}
                    </div>
                    <div className="text-[10px] text-slate-500 truncate">{a.authors}{a.description ? ` — ${a.description}` : ''}</div>
                  </div>
                  <button onClick={e => { e.stopPropagation(); onInfo(a.name) }}
                    className="flex-shrink-0 w-5 h-5 rounded-full bg-slate-700/60 hover:bg-[var(--accent)]/30 flex items-center justify-center text-[10px] text-slate-400 hover:text-[var(--accent)] transition-colors font-serif italic"
                    title="Info">
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


/* ── Results display ─────────────────────────────── */
function TextAttackResult({ result, attack, model }) {
  const r = result
  const success = r.success

  return (
    <Card className={success ? 'border-green-500/30' : 'border-amber-500/30'}>
      <SectionLabel>Results</SectionLabel>

      {/* Status badge */}
      <div className="flex items-center gap-3 mb-4">
        <span className={`text-lg font-bold ${success ? 'text-green-400' : 'text-amber-400'}`}>
          {success ? '✓ Attack Successful' : '✗ Attack Failed'}
        </span>
        <span className="text-xs text-slate-500">{attack} | {model}</span>
        {r.elapsed_ms && <span className="text-xs text-slate-600">{(r.elapsed_ms / 1000).toFixed(1)}s</span>}
      </div>

      {/* Side-by-side text comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <TextCard label="Original" text={r.original_text} labelName={r.original_label} confidence={r.original_confidence} color="blue" />
        <TextCard label="Adversarial" text={r.adversarial_text} labelName={r.adversarial_label} confidence={r.adversarial_confidence} color={success ? 'green' : 'amber'} />
      </div>

      {/* Diff view — highlight changed words */}
      {r.original_text && r.adversarial_text && r.original_text !== r.adversarial_text && (
        <div className="mb-6">
          <span className="block text-xs text-slate-500 mb-2">Changed Words</span>
          <DiffView original={r.original_text} adversarial={r.adversarial_text} />
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <MetricBox label="Perturbation" value={r.perturbation_ratio != null ? `${(r.perturbation_ratio * 100).toFixed(1)}%` : '—'} />
        <MetricBox label="Semantic Sim." value={r.semantic_similarity != null ? `${(r.semantic_similarity * 100).toFixed(1)}%` : '—'} />
        <MetricBox label="Queries" value={r.num_queries || '—'} />
        <MetricBox label="Orig → Adv" value={`${r.original_label || '?'} → ${r.adversarial_label || '?'}`} />
      </div>

      {/* Prediction bars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {r.original_predictions && <PredBars label="Original predictions" preds={r.original_predictions} highlight={r.original_label} color="blue" />}
        {r.adversarial_predictions && <PredBars label="Adversarial predictions" preds={r.adversarial_predictions} highlight={r.adversarial_label} color="green" />}
      </div>
    </Card>
  )
}


/* ── Small UI primitives ─────────────────────────── */
function TextCard({ label, text, labelName, confidence, color }) {
  const borderColor = color === 'green' ? 'border-green-500/30' : color === 'amber' ? 'border-amber-500/30' : 'border-blue-500/30'
  const textColor   = color === 'green' ? 'text-green-400'      : color === 'amber' ? 'text-amber-400'      : 'text-blue-400'
  return (
    <div className={`rounded-lg border ${borderColor} bg-slate-900/40 p-4`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</span>
        {labelName && (
          <span className={`text-xs font-semibold ${textColor}`} title={`Confidence: ${(confidence * 100).toFixed(1)}%`}>
            {labelName} ({(confidence * 100).toFixed(1)}%)
          </span>
        )}
      </div>
      <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap break-words">{text}</p>
    </div>
  )
}

function DiffView({ original, adversarial }) {
  const origWords = original.split(/\s+/)
  const advWords  = adversarial.split(/\s+/)
  const maxLen = Math.max(origWords.length, advWords.length)

  return (
    <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-3 text-sm leading-relaxed">
      {Array.from({ length: maxLen }).map((_, i) => {
        const ow = origWords[i] || ''
        const aw = advWords[i] || ''
        if (ow === aw) return <span key={i} className="text-slate-400">{aw} </span>
        return (
          <span key={i}>
            {ow && <span className="bg-red-500/20 text-red-300 line-through rounded px-0.5 mx-0.5">{ow}</span>}
            {aw && <span className="bg-emerald-500/20 text-emerald-300 rounded px-0.5 mx-0.5">{aw}</span>}
            {' '}
          </span>
        )
      })}
    </div>
  )
}

function MetricBox({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-800/60 border border-slate-700/50 p-3 text-center">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-sm font-semibold text-slate-200">{value}</div>
    </div>
  )
}

function PredBars({ label, preds, highlight, color }) {
  if (!preds) return null
  // preds can be either {label: conf, ...} or [{label, score}, ...]
  let entries
  if (Array.isArray(preds)) {
    entries = preds.map(p => [p.label, p.score || p.confidence || 0])
  } else {
    entries = Object.entries(preds)
  }
  const c  = color === 'green' ? 'bg-green-500/60' : 'bg-blue-500/60'
  const ch = color === 'green' ? 'bg-green-400'     : 'bg-blue-400'
  return (
    <div className="mb-4">
      <p className="text-xs text-slate-500 mb-2">{label}</p>
      <div className="space-y-1.5">
        {entries.slice(0, 5).map(([k, v]) => (
          <div key={k} className="flex items-center gap-2">
            <div className="w-3/5 bg-slate-700/80 rounded-full h-3.5 overflow-hidden">
              <div className={`h-full rounded-full transition-all ${k === highlight ? ch : c}`} style={{ width: `${Math.min(v * 100, 100)}%` }} />
            </div>
            <span className={`text-[11px] truncate ${k === highlight ? 'text-slate-200 font-medium' : 'text-slate-400'}`}>{k}: {(v * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
