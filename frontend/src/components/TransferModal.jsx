import { useState, useEffect, useCallback } from 'react'
import {
  getImageVerificationStatus,
  getImageHeartbeat,
  getVerificationTargets,
  runImageVerification,
  getEnabledPlugins,
} from '../api/client'

const DOT = { ready: 'bg-green-400', degraded: 'bg-amber-400', unavailable: 'bg-red-400', checking: 'bg-slate-400 animate-pulse' }

function normalizeTargetItems(items = []) {
  return items.map(item => ({
    ...item,
    label: item.label || item.display_name || item.value || item.id,
    value: item.value || item.id || item.model_ref,
  })).filter(item => item.value)
}

function StatusDot({ level }) {
  return <span className={`inline-block w-2 h-2 rounded-full ${DOT[level] || DOT.unavailable}`} />
}

function targetServiceName(target) {
  return target?.settings?.service_name || target?.display_name || target?.label || target?.value
}

function formatProvider(provider) {
  const normalized = String(provider || '').trim().toLowerCase()
  if (!normalized) return ''
  if (normalized === 'gcp') return 'Google Cloud'
  if (normalized === 'aws') return 'AWS'
  if (normalized === 'azure') return 'Azure'
  if (normalized === 'hf' || normalized === 'huggingface') return 'Hugging Face'
  return provider
}

function targetSourceSummary(target) {
  const service = target?.settings?.service_name || target?.backend || 'remote'
  const provider = formatProvider(target?.provider)
  return provider ? `${service} · ${provider}` : service
}

export default function TransferModal({ adversarialB64, originalB64, targetLabel, originalLabel, onClose }) {
  const [services, setServices] = useState([])
  const [remoteTargets, setRemoteTargets] = useState([])
  const [localTargets, setLocalTargets] = useState([])

  const [selectedRemoteTargets, setSelectedRemoteTargets] = useState([])
  const [selectedLocalModels, setSelectedLocalModels] = useState([])
  const [localExpanded, setLocalExpanded] = useState(false)
  const [preprocessMode, setPreprocessMode] = useState('exact')

  const [imagePlugins, setImagePlugins] = useState([])
  const [selectedPlugins, setSelectedPlugins] = useState([])

  const [heartbeats, setHeartbeats] = useState({})
  const [hbLoading, setHbLoading] = useState(true)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let mounted = true
    const init = async () => {
      try {
        const [statusData, targetData, pluginsData] = await Promise.all([
          getImageVerificationStatus(),
          getVerificationTargets('image'),
          getEnabledPlugins('image'),
        ])
        if (!mounted) return
        const nextServices = statusData.services || []
        const nextRemoteTargets = normalizeTargetItems(targetData.targets || [])
        const nextLocalTargets = normalizeTargetItems(targetData.local_targets || [])
        setServices(nextServices)
        setRemoteTargets(nextRemoteTargets)
        setLocalTargets(nextLocalTargets)
        if (nextLocalTargets.length > 0) {
          setSelectedLocalModels(prev => prev.length ? prev : [nextLocalTargets[0].value])
        }
        setImagePlugins(pluginsData.plugins || [])

        const autoSelected = nextRemoteTargets
          .filter(target => {
            const status = nextServices.find(service => service.name === targetServiceName(target))
            return status && (status.level === 'ready' || status.level === 'degraded')
          })
          .map(target => target.value)
        setSelectedRemoteTargets(autoSelected)
      } catch (e) {
        console.error('Failed to load status', e)
      }

      try {
        setHbLoading(true)
        const hbData = await getImageHeartbeat()
        if (!mounted) return
        const map = {}
        for (const s of (hbData.services || [])) map[s.name] = s
        setHeartbeats(map)
      } catch (e) {
        console.error('Heartbeat failed', e)
      } finally {
        if (mounted) setHbLoading(false)
      }
    }
    init()
    return () => { mounted = false }
  }, [])

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose?.()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  const refreshHeartbeat = useCallback(async () => {
    setHbLoading(true)
    try {
      const hbData = await getImageHeartbeat()
      const map = {}
      for (const s of (hbData.services || [])) map[s.name] = s
      setHeartbeats(map)
    } catch (e) {
      console.error(e)
    } finally {
      setHbLoading(false)
    }
  }, [])

  const toggleRemoteTarget = (targetId) => {
    setSelectedRemoteTargets(prev => prev.includes(targetId) ? prev.filter(id => id !== targetId) : [...prev, targetId])
  }

  const toggleLocalModel = (name) => {
    setSelectedLocalModels(prev =>
      prev.includes(name) ? prev.filter(m => m !== name) : [...prev, name]
    )
  }

  const togglePlugin = (id) => {
    setSelectedPlugins(prev =>
      prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]
    )
  }

  const handleRun = async () => {
    setLoading(true)
    setError('')
    setResults(null)
    try {
      const data = await runImageVerification({
        adversarial_b64: adversarialB64,
        original_b64: originalB64 || null,
        target_label: targetLabel,
        original_label: originalLabel,
        remote_target_ids: selectedRemoteTargets.length > 0 ? selectedRemoteTargets : null,
        local_model_ids: selectedLocalModels.length > 0 ? selectedLocalModels : null,
        preprocess_mode: preprocessMode,
        plugin_ids: selectedPlugins.length > 0 ? selectedPlugins : null,
      })
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const getLevel = (target) => {
    const name = targetServiceName(target)
    if (hbLoading) return 'checking'
    const hb = heartbeats[name]
    if (!hb) {
      const svc = services.find(s => s.name === name)
      return svc?.level || 'unavailable'
    }
    if (hb.ok) return hb.level || 'ready'
    return 'unavailable'
  }

  const getReasonText = (target) => {
    const name = targetServiceName(target)
    if (hbLoading) return 'Checking...'
    const hb = heartbeats[name]
    if (hb) return hb.message || hb.reason || ''
    const svc = services.find(s => s.name === name)
    return svc?.reason || svc?.status || ''
  }

  const isRemoteTargetSelectable = (target) => getLevel(target) !== 'unavailable'

  useEffect(() => {
    if (hbLoading) return
    setSelectedRemoteTargets(prev => prev.filter(targetId => {
      const target = remoteTargets.find(item => item.value === targetId)
      return target ? isRemoteTargetSelectable(target) : false
    }))
  }, [hbLoading, heartbeats, services, remoteTargets])

  const canRun = selectedRemoteTargets.length > 0 || selectedLocalModels.length > 0 || selectedPlugins.length > 0

  const localLabel = value => localTargets.find(model => model.value === value)?.label || value

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-slate-800 border border-slate-600 rounded-2xl p-6 max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl" onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-lg font-semibold text-[var(--accent)]">Transfer Verification</h3>
            <p className="text-xs text-slate-400 mt-1">Test adversarial image against external models to verify attack transfer.</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-red-400 text-xl leading-none px-2 py-1 rounded transition">×</button>
        </div>

        {/* Target Cards */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium text-slate-300">Remote Targets</p>
            <button onClick={refreshHeartbeat} disabled={hbLoading}
              className="text-xs text-slate-400 hover:text-[var(--accent)] transition flex items-center gap-1 disabled:opacity-50">
              <svg className={`w-3 h-3 ${hbLoading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {remoteTargets.map(target => {
              const level = getLevel(target)
              const reason = getReasonText(target)
              const isSelected = selectedRemoteTargets.includes(target.value)
              const isDisabled = !isRemoteTargetSelectable(target)
              return (
                <button
                  key={target.value}
                  type="button"
                  disabled={isDisabled}
                  onClick={() => toggleRemoteTarget(target.value)}
                  className={`text-left p-3 rounded-lg border transition-all ${
                    isSelected
                      ? 'border-[var(--accent)] bg-[var(--accent)]/5'
                      : 'border-slate-700 hover:border-slate-500'
                  } ${isDisabled ? 'opacity-60 cursor-not-allowed hover:border-slate-700' : ''}`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <input type="checkbox" checked={isSelected} readOnly className="pointer-events-none" />
                    <StatusDot level={level} />
                    <span className="text-sm font-medium text-slate-200">{target.label}</span>
                  </div>
                  <p className="text-[10px] ml-6 text-slate-500 mb-0.5">
                    {targetSourceSummary(target)}
                  </p>
                  <p className={`text-[10px] ml-6 ${level === 'ready' ? 'text-green-400' : level === 'degraded' ? 'text-amber-400' : level === 'checking' ? 'text-slate-400' : 'text-red-400'}`}>
                    {reason}
                  </p>
                </button>
              )
            })}
          </div>
        </div>

        {/* Plugins Section */}
        {imagePlugins.length > 0 && (
          <div className="mb-4">
            <p className="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
              Plugins
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 font-normal">
                {imagePlugins.length} available
              </span>
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {imagePlugins.map(p => {
                const isOn = selectedPlugins.includes(p.id)
                return (
                  <button
                    key={p.id}
                    onClick={() => togglePlugin(p.id)}
                    className={`text-left p-3 rounded-lg border transition-all ${
                      isOn
                        ? 'border-[var(--accent)] bg-[var(--accent)]/5'
                        : 'border-slate-700 hover:border-slate-500'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <input type="checkbox" checked={isOn} readOnly className="pointer-events-none" />
                      <span className="w-2 h-2 rounded-full flex-shrink-0 bg-amber-400" />
                      <span className="text-sm font-medium text-slate-200 truncate">{p.name}</span>
                    </div>
                    <p className="text-[10px] ml-6 text-slate-500">
                      Local Python
                      {p.description ? ` · ${p.description}` : ''}
                    </p>
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* Local Model Picker */}
        {localTargets.length > 0 && (
          <div className="mb-4 p-3 bg-slate-900/50 rounded-lg border border-slate-700/50">
            <button onClick={() => setLocalExpanded(e => !e)} className="flex items-center gap-2 text-sm text-slate-300 w-full">
              <svg className={`w-3 h-3 transition-transform ${localExpanded ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20"><path d="M6 4l8 6-8 6V4z" /></svg>
              <span className="font-medium">Local Models</span>
              <span className="text-[10px] bg-[var(--accent)]/20 text-[var(--accent)] px-1.5 py-0.5 rounded-full">
                {selectedLocalModels.length} selected
              </span>
              {selectedLocalModels.length > 0 && (
                <button onClick={e => { e.stopPropagation(); setSelectedLocalModels([]) }}
                  className="text-[10px] text-slate-500 hover:text-red-400 ml-auto">Clear</button>
              )}
            </button>
            {localExpanded && (
              <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-1.5 max-h-52 overflow-y-auto p-1">
                {localTargets.map(model => (
                  <label key={model.value} className={`flex items-center gap-1.5 px-2 py-1.5 rounded text-xs cursor-pointer transition ${
                    selectedLocalModels.includes(model.value) ? 'bg-[var(--accent)]/10 text-[var(--accent)]' : 'text-slate-400 hover:bg-slate-800'
                  }`}>
                    <input type="checkbox" checked={selectedLocalModels.includes(model.value)} onChange={() => toggleLocalModel(model.value)} className="w-3 h-3" />
                    <span className="truncate">{model.label}</span>
                  </label>
                ))}
              </div>
            )}
            {!localExpanded && selectedLocalModels.length > 0 && (
              <p className="text-[10px] text-slate-500 mt-1 ml-5 truncate">
                {selectedLocalModels.slice(0, 3).map(localLabel).join(', ')}{selectedLocalModels.length > 3 ? ` +${selectedLocalModels.length - 3} more` : ''}
              </p>
            )}
          </div>
        )}

        {/* Local Preprocessing Mode */}
        {localTargets.length > 0 && (
          <div className="mb-4 p-3 bg-slate-900/50 rounded-lg border border-slate-700/50">
            <p className="text-sm font-medium text-slate-300 mb-2">Local Preprocessing</p>
            <div className="flex items-start gap-4">
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="radio" name="preprocess_mode" checked={preprocessMode === 'exact'} onChange={() => setPreprocessMode('exact')} className="mt-0.5" />
                <div>
                  <span className="text-xs text-slate-200 font-medium">Exact (pixel-perfect)</span>
                  <p className="text-[10px] text-slate-500 mt-0.5">Bypass resize/crop — feeds adversarial pixels directly. True transfer test.</p>
                </div>
              </label>
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="radio" name="preprocess_mode" checked={preprocessMode === 'standard'} onChange={() => setPreprocessMode('standard')} className="mt-0.5" />
                <div>
                  <span className="text-xs text-slate-200 font-medium">Standard (resize + crop)</span>
                  <p className="text-[10px] text-slate-500 mt-0.5">Full processor pipeline — simulates real-world deployment where the receiver resizes.</p>
                </div>
              </label>
            </div>
          </div>
        )}

        {/* Labels */}
        <div className="flex gap-4 mb-4 text-xs text-slate-400">
          {targetLabel && <span>Target: <span className="text-[var(--accent)]">{targetLabel}</span></span>}
          {originalLabel && <span>Original: <span className="text-slate-300">{originalLabel}</span></span>}
        </div>

        {/* Run Button */}
        <button
          onClick={handleRun}
          disabled={loading || !canRun}
          className="px-6 py-3 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-semibold rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? 'Running…' : 'Run Transfer Test'}
        </button>

        {error && <p className="mt-4 text-red-400 text-sm">{error}</p>}

        {/* Results */}
        {results && (
          <div className="mt-6 border-t border-slate-600 pt-6">
            <h4 className="text-sm font-semibold text-[var(--accent)] mb-4">Results</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-600">
                    <th className="text-left py-2 pr-3">Service</th>
                    <th className="text-left py-2 pr-3">Transfer</th>
                    <th className="text-left py-2 pr-3">Top Predictions</th>
                    <th className="text-left py-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results?.map(r => (
                    <tr key={r.verifier_name} className="border-b border-slate-700/50">
                      <td className="py-3 pr-3">
                        <span className="font-medium text-slate-200">{r.verifier_name}</span>
                        <br />
                        <span className={`text-[10px] ${r.service_type === 'plugin' ? 'text-cyan-400' : 'text-slate-500'}`}>
                          {r.service_type === 'plugin' ? '⚡ plugin' : r.service_type}
                        </span>
                      </td>
                      <td className="py-3 pr-3">
                        {r.error ? (
                          <span className="text-red-400 text-xs">{r.error}</span>
                        ) : r.matched_target ? (
                          <span className="text-green-400 font-semibold text-xs">TARGET MATCHED</span>
                        ) : r.original_label_gone ? (
                          <span className="text-amber-400 font-medium text-xs">ORIGINAL GONE</span>
                        ) : (
                          <span className="text-slate-500 text-xs">NO EFFECT</span>
                        )}
                        {!r.error && r.confidence_drop > 0 && (
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Confidence drop: {(r.confidence_drop * 100).toFixed(1)}%
                          </div>
                        )}
                      </td>
                      <td className="py-3 pr-3">
                        {r.predictions?.length ? r.predictions.map((p, i) => (
                          <div key={i} className="text-xs text-slate-300">
                            <span className="text-slate-400">{p.label}:</span> {(p.confidence * 100).toFixed(1)}%
                          </div>
                        )) : <span className="text-slate-500 text-xs">—</span>}
                      </td>
                      <td className="py-3 text-xs text-slate-400">
                        {r.elapsed_ms ? `${r.elapsed_ms.toFixed(0)}ms` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
