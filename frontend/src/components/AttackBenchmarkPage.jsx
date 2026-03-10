import { useState, useEffect, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { getBenchmarkAttacks, getModels, getAsrModels, getVerificationTargets, runBenchmarkSSE } from '../api/client'
import BenchmarkConfig from './benchmark/BenchmarkConfig'
import BenchmarkResults from './benchmark/BenchmarkResults'

function normalizeModelItems(response) {
  if (Array.isArray(response?.items)) {
    return response.items.map(item => ({
      label: item.label || item.display_name || item.value || item.id,
      value: item.value || item.id || item.model_ref,
      group: item.group || item.family || item.task || 'Other',
    })).filter(item => item.value)
  }
  return (response?.models || []).map(value => ({ label: value, value, group: 'Other' }))
}

function normalizeTargetItems(items = []) {
  return items.map(item => ({
    ...item,
    label: item.label || item.display_name || item.value || item.id,
    value: item.value || item.id || item.model_ref,
    group: item.group || item.family || item.task || 'Other',
  })).filter(item => item.value)
}

export default function AttackBenchmarkPage() {
  const [domain, setDomain] = useState('image')
  const [imageAttackGroups, setImageAttackGroups] = useState({})
  const [audioAttacks, setAudioAttacks] = useState([])
  const [allModels, setAllModels] = useState([])
  const [asrModels, setAsrModels] = useState([])
  const [imageLocalTargets, setImageLocalTargets] = useState([])
  const [imageRemoteTargets, setImageRemoteTargets] = useState([])
  const [audioRemoteTargets, setAudioRemoteTargets] = useState([])

  const [running, setRunning] = useState(false)
  const [results, setResults] = useState([])
  const [summary, setSummary] = useState(null)
  const [total, setTotal] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const [completed, setCompleted] = useState(false)
  const [currentJobId, setCurrentJobId] = useState('')
  const abortRef = useRef(null)
  const timerRef = useRef(null)
  const resultsRef = useRef(null)
  const scrolledRef = useRef(false)

  useEffect(() => {
    getBenchmarkAttacks().then(d => {
      setImageAttackGroups(d.image || {})
      setAudioAttacks(d.audio || [])
    }).catch(() => {})
    getModels().then(d => {
      setAllModels(normalizeModelItems(d))
    }).catch(() => {})
    getAsrModels().then(d => {
      setAsrModels(normalizeModelItems(d))
    }).catch(() => {})
    getVerificationTargets('image').then(d => {
      setImageLocalTargets(normalizeTargetItems(d.local_targets || []))
      setImageRemoteTargets(normalizeTargetItems(d.targets || []))
    }).catch(() => {})
    getVerificationTargets('audio').then(d => {
      setAudioRemoteTargets(normalizeTargetItems(d.targets || []))
    }).catch(() => {})
  }, [])

  // Auto-scroll to results when first result arrives
  useEffect(() => {
    if (results.length === 1 && !scrolledRef.current && resultsRef.current) {
      scrolledRef.current = true
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [results.length])

  // Auto-scroll to results header when benchmark completes
  useEffect(() => {
    if (completed && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [completed])

  // Auto-dismiss completion banner after 8s
  useEffect(() => {
    if (!completed) return
    const t = setTimeout(() => setCompleted(false), 8000)
    return () => clearTimeout(t)
  }, [completed])

  const handleRun = useCallback(fd => {
    setResults([])
    setSummary(null)
    setRunning(true)
    setElapsed(0)
    setError('')
    setCompleted(false)
    setCurrentJobId('')
    scrolledRef.current = false

    const t0 = Date.now()
    clearInterval(timerRef.current)
    timerRef.current = setInterval(() => setElapsed(Date.now() - t0), 500)

    const handle = runBenchmarkSSE(fd, {
      onCreated: job => setCurrentJobId(job.job_id),
      onInit: data => setTotal(data.total || 0),
      onResult: data => setResults(prev => [...prev, data]),
      onSummary: data => setSummary(data),
      onDone: () => {
        setRunning(false)
        setCompleted(true)
        clearInterval(timerRef.current)
      },
      onError: err => {
        setRunning(false)
        clearInterval(timerRef.current)
        setError(err.message === 'Job cancelled' ? 'Job cancelled. Open Jobs to resume later.' : (err.message || 'Benchmark failed - check server logs'))
      },
    })
    abortRef.current = handle
  }, [])

  const handleAbort = () => {
    abortRef.current?.abort()
    setRunning(false)
    clearInterval(timerRef.current)
    setError('Cancellation requested. Open Jobs to monitor or resume later.')
  }

  const resultCount = results.length
  const hasResults = resultCount > 0

  return (
    <div className="space-y-6 pb-20">
      <BenchmarkConfig
        domain={domain} setDomain={setDomain}
        imageAttackGroups={imageAttackGroups}
        audioAttacks={audioAttacks}
        allModels={allModels}
        asrModels={asrModels}
        imageLocalTargets={imageLocalTargets}
        imageRemoteTargets={imageRemoteTargets}
        audioRemoteTargets={audioRemoteTargets}
        onRun={handleRun}
        loading={running}
      />

      {currentJobId && (
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <p className="text-sm font-semibold text-slate-200">Benchmark is running as a durable background job.</p>
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
            Use Jobs for persistent status, cancellation, and resume across refreshes.
          </p>
        </div>
      )}

      {/* Progress bar while running */}
      {running && (
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
              <span className="text-sm font-semibold text-slate-200">
                Running benchmark… {resultCount}/{total}
              </span>
              <span className="text-[10px] text-slate-500">{(elapsed / 1000).toFixed(0)}s</span>
            </div>
            <button onClick={handleAbort}
              className="px-3 py-1 rounded-lg text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition">
              Abort
            </button>
          </div>
          <div className="w-full bg-slate-700/40 rounded-full h-2 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent)] rounded-full transition-all duration-300"
              style={{ width: `${total ? (resultCount / total) * 100 : 0}%` }} />
          </div>
          {total > resultCount && resultCount > 0 && (
            <p className="text-[10px] text-slate-600 mt-1.5">
              ~{(((elapsed / resultCount) * (total - resultCount)) / 1000).toFixed(0)}s remaining
            </p>
          )}
        </div>
      )}

      {/* Completion notification */}
      {completed && !running && hasResults && (
        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4 flex items-center justify-between animate-[fadeIn_0.3s_ease-out]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center shrink-0">
              <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <span className="text-sm font-semibold text-emerald-300">Benchmark Complete</span>
              <span className="text-xs text-emerald-400/70 ml-2">
                {resultCount} combination{resultCount !== 1 ? 's' : ''} tested in {(elapsed / 1000).toFixed(1)}s
                {summary?.best_attack && <> — Best: <span className="font-medium text-emerald-300">{summary.best_attack}</span> ({((summary.best_transfer_rate || 0) * 100).toFixed(0)}% transfer)</>}
              </span>
            </div>
          </div>
          <button onClick={() => setCompleted(false)} className="text-emerald-500/50 hover:text-emerald-400 transition p-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-center justify-between">
          <span className="text-sm text-red-400">{error}</span>
          <button onClick={() => setError('')} className="text-red-500/50 hover:text-red-400 transition p-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      )}

      {/* Results section */}
      {hasResults && (
        <div ref={resultsRef} className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Benchmark Results</h3>
              <p className="text-[10px] text-slate-500">
                {resultCount} combination{resultCount !== 1 ? 's' : ''} tested · {(elapsed / 1000).toFixed(1)}s total
                {running && <span className="text-[var(--accent)] ml-1">(streaming…)</span>}
              </p>
            </div>
            {summary?.best_attack && !running && (
              <span className="text-[10px] text-slate-400">
                Best: <span className="text-[var(--accent)] font-medium">{summary.best_attack}</span> — {((summary.best_transfer_rate || 0) * 100).toFixed(0)}% transfer
              </span>
            )}
          </div>
          <BenchmarkResults
            results={results}
            summary={summary}
            total={total}
            elapsed={elapsed}
            domain={domain}
            running={running}
            onAbort={handleAbort}
          />
        </div>
      )}
    </div>
  )
}
