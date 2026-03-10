import { useState, useEffect, useCallback } from 'react'
import { getAudioVerificationStatus, getAudioHeartbeat, getVerificationTargets, runAudioVerification, getEnabledPlugins } from '../api/client'

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

function normalizeText(value) {
  return String(value || '').trim().toLowerCase()
}

function baselineText(result, fallbackOriginalText) {
  return result?.original_transcription || fallbackOriginalText || ''
}

function buildResultStatus(result, targetedMode, fallbackOriginalText) {
  if (result.error) {
    return { label: result.error, className: 'text-red-400 text-xs' }
  }

  if (targetedMode) {
    if (result.exact_match) return { label: 'EXACT MATCH', className: 'text-green-400 font-semibold text-xs' }
    if (result.contains_target) return { label: 'CONTAINS TARGET', className: 'text-amber-400 font-medium text-xs' }
    return { label: 'NO MATCH', className: 'text-slate-500 text-xs' }
  }

  const baseline = baselineText(result, fallbackOriginalText)
  const changed = (normalizeText(result.transcription) !== normalizeText(baseline)) && Boolean(result.transcription || baseline)
  return {
    label: changed ? 'CHANGED' : 'UNCHANGED',
    className: changed ? 'text-green-400 font-semibold text-xs' : 'text-slate-500 text-xs',
  }
}

export default function AudioTransferModal({ adversarialWavB64, originalWavB64, sampleRate, targetText, originalText, language = 'en-US', onClose }) {
  const [services, setServices] = useState([])
  const [targets, setTargets] = useState([])
  const [selectedTargets, setSelectedTargets] = useState([])
  const [heartbeats, setHeartbeats] = useState({})
  const [hbLoading, setHbLoading] = useState(true)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  const [audioPlugins, setAudioPlugins] = useState([])
  const [selectedPlugins, setSelectedPlugins] = useState([])
  const targetedMode = Boolean(targetText?.trim())

  useEffect(() => {
    let mounted = true
    const init = async () => {
      try {
        const [statusResult, targetResult, pluginsResult] = await Promise.allSettled([
          getAudioVerificationStatus(),
          getVerificationTargets('audio'),
          getEnabledPlugins('audio'),
        ])
        if (!mounted) return
        const statusData = statusResult.status === 'fulfilled' ? statusResult.value : {}
        const targetData = targetResult.status === 'fulfilled' ? targetResult.value : {}
        const pluginsData = pluginsResult.status === 'fulfilled' ? pluginsResult.value : {}
        const list = Array.isArray(statusData) ? statusData : (statusData.services || [])
        const nextTargets = normalizeTargetItems(targetData.targets || [])
        setServices(list)
        setTargets(nextTargets)
        setAudioPlugins(pluginsData.plugins || [])
        const autoSelected = nextTargets
          .filter(target => {
            const status = list.find(service => service.name === targetServiceName(target))
            return status && (status.level === 'ready' || status.level === 'degraded')
          })
          .map(target => target.value)
        setSelectedTargets(autoSelected)
      } catch (e) {
        console.error('Failed to load audio status', e)
      }

      try {
        setHbLoading(true)
        const hbData = await getAudioHeartbeat()
        if (!mounted) return
        const map = {}
        for (const s of (hbData.services || [])) map[s.name] = s
        setHeartbeats(map)
      } catch (e) {
        console.error('Audio heartbeat failed', e)
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
      const hbData = await getAudioHeartbeat()
      const map = {}
      for (const s of (hbData.services || [])) map[s.name] = s
      setHeartbeats(map)
    } catch (e) {
      console.error(e)
    } finally {
      setHbLoading(false)
    }
  }, [])

  const toggleTarget = (targetId) => {
    setSelectedTargets(prev => prev.includes(targetId) ? prev.filter(id => id !== targetId) : [...prev, targetId])
  }

  const togglePlugin = (id) => {
    setSelectedPlugins(prev => prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id])
  }

  const getLevel = (target) => {
    const name = targetServiceName(target)
    if (hbLoading) return 'checking'
    const hb = heartbeats[name]
    if (!hb) {
      const svc = services.find(s => s.name === name)
      return svc?.level || 'unavailable'
    }
    return hb.ok ? (hb.level || 'ready') : 'unavailable'
  }

  const getReasonText = (target) => {
    const name = targetServiceName(target)
    if (hbLoading) return 'Checking...'
    const hb = heartbeats[name]
    if (hb) return hb.message || hb.reason || ''
    const svc = services.find(s => s.name === name)
    return svc?.reason || svc?.status || ''
  }

  const isTargetSelectable = (target) => getLevel(target) !== 'unavailable'

  useEffect(() => {
    if (hbLoading) return
    setSelectedTargets(prev => prev.filter(targetId => {
      const target = targets.find(item => item.value === targetId)
      return target ? isTargetSelectable(target) : false
    }))
  }, [hbLoading, heartbeats, services, targets])

  const handleRun = async () => {
    setLoading(true)
    setError('')
    setResults(null)
    try {
      const data = await runAudioVerification({
        adversarial_b64: adversarialWavB64,
        original_b64: originalWavB64,
        sample_rate: sampleRate,
        target_text: targetText,
        original_text: originalText,
        remote_target_ids: selectedTargets.length > 0 ? selectedTargets : null,
        language,
        plugin_ids: selectedPlugins.length > 0 ? selectedPlugins : null,
      })
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const canRun = selectedTargets.length > 0 || selectedPlugins.length > 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-slate-800 border border-slate-600 rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl" onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-lg font-semibold text-[var(--accent)]">Audio Transfer Verification</h3>
            <p className="text-xs text-slate-400 mt-1">Test adversarial audio on external ASR services.</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-red-400 text-xl px-2 py-1 rounded transition">×</button>
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

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {targets.map(target => {
              const level = getLevel(target)
              const reason = getReasonText(target)
              const isSelected = selectedTargets.includes(target.value)
              const isDisabled = !isTargetSelectable(target)
              return (
                <button
                  key={target.value}
                  type="button"
                  disabled={isDisabled}
                  onClick={() => toggleTarget(target.value)}
                  className={`text-left p-3 rounded-lg border transition-all ${
                    isSelected ? 'border-[var(--accent)] bg-[var(--accent)]/5' : 'border-slate-700 hover:border-slate-500'
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

        {/* Audio Plugins Section */}
        {audioPlugins.length > 0 && (
          <div className="mb-4">
            <p className="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
              Plugins
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 font-normal">
                {audioPlugins.length} available
              </span>
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {audioPlugins.map(p => {
                const isOn = selectedPlugins.includes(p.id)
                return (
                  <button
                    key={p.id}
                    onClick={() => togglePlugin(p.id)}
                    className={`text-left p-3 rounded-lg border transition-all ${
                      isOn ? 'border-[var(--accent)] bg-[var(--accent)]/5' : 'border-slate-700 hover:border-slate-500'
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

        {/* Target info */}
        <div className="flex gap-4 mb-4 text-xs text-slate-400">
          {targetText && <span>Target: <span className="text-[var(--accent)]">"{targetText}"</span></span>}
          {originalText && <span>Original: <span className="text-slate-300">"{originalText}"</span></span>}
        </div>

        {/* Run */}
        <button onClick={handleRun} disabled={loading || !canRun}
          className="px-6 py-3 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-semibold rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition">
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
                    <th className="text-left py-2 pr-3">Result</th>
                    <th className="text-left py-2 pr-3">Transcription</th>
                    <th className="text-left py-2 pr-3">Baseline</th>
                    <th className="text-left py-2 pr-3">WER</th>
                    <th className="text-left py-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results?.map(r => {
                    const status = buildResultStatus(r, targetedMode, originalText)
                    const baseline = baselineText(r, originalText)
                    return (
                    <tr key={r.verifier_name} className="border-b border-slate-700/50">
                      <td className="py-3 pr-3">
                        <span className="font-medium text-slate-200">{r.verifier_name}</span>
                        <br />
                        <span className={`text-[10px] ${r.service_type === 'plugin' ? 'text-cyan-400' : 'text-slate-500'}`}>
                          {r.service_type === 'plugin' ? '⚡ plugin' : r.service_type}
                        </span>
                      </td>
                      <td className="py-3 pr-3"><span className={status.className}>{status.label}</span></td>
                      <td className="py-3 pr-3 text-xs text-slate-300 max-w-xs truncate">{r.transcription || '—'}</td>
                      <td className="py-3 pr-3 text-xs text-slate-500 max-w-xs truncate">{baseline || '—'}</td>
                      <td className="py-3 pr-3 text-xs text-slate-400">{r.wer != null && !r.error ? `${(r.wer * 100).toFixed(0)}%` : '—'}</td>
                      <td className="py-3 text-xs text-slate-400">{r.elapsed_ms ? `${r.elapsed_ms.toFixed(0)}ms` : '—'}</td>
                    </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
