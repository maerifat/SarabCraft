import { useState, useEffect, useMemo } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { runImageAttack, followImageAttackJob, getModels, getAttacks, createAbortable, cancelJobById, getJobDetails, getJobArtifactDataUrl } from '../api/client'
import TransferModal from './TransferModal'
import MetricsPanel from './MetricsPanel'
import GradCAMPanel from './GradCAMPanel'
import Slider from './ui/Slider'
import { Card, SectionLabel, Row, Select, RunButton, ErrorMsg } from './ui/Section'
import ModelPicker from './ui/ModelPicker'
import { FALLBACK_REGISTRY, THREAT_COLORS, THREAT_LABEL } from './image/attackRegistry'
import AttackInfoModal from './image/AttackInfoModal'
import ParamRenderer from './image/ParamRenderer'
import EnsemblePanel from './image/EnsemblePanel'
import { buildAttackParamPayload } from './image/sarabcraftR1'
import { downloadB64, slugify } from '../utils/download'
import { downloadReport, downloadJSON } from '../utils/report'
import { useImageAttack } from './ImageAttackContext'

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

export default function ImageAttackTab() {
  /* ── Persistent state (survives navigation via Context) ───────── */
  const {
    result, setResult,
    inputFile, setInputFile,
    inputPreview, setInputPreview,
    targetFile, setTargetFile,
    targetPreview, setTargetPreview,
    model, setModel,
    attack, setAttack,
    paramValues, setParamValues,
    ensembleModels, setEnsembleModels,
    ensembleMode, setEnsembleMode,
    loading, setLoading,
    error, setError,
    currentJobId, setCurrentJobId,
    progress, setProgress,
    abortRef,
    jobIdRef,
    restoreFromJob,
  } = useImageAttack()

  /* ── Ephemeral local state (OK to reset on remount) ───────────── */
  const [modelsList, setModelsList] = useState([])
  const [registry, setRegistry] = useState(FALLBACK_REGISTRY)
  const [transferModal, setTransferModal] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [infoOpen, setInfoOpen] = useState(null)
  const [searchParams, setSearchParams] = useSearchParams()

  useEffect(() => {
    Promise.all([getModels(), getAttacks()]).then(([m, a]) => {
      const items = normalizeModelItems(m)
      setModelsList(items)
      if (items.length > 0) setModel(prev => prev || m.default_id || items[0].value)
      if (a.registry && Object.keys(a.registry).length > 0) {
        setRegistry(a.registry)
      } else {
        setRegistry(FALLBACK_REGISTRY)
      }
    }).catch(() => setRegistry(FALLBACK_REGISTRY))
  }, [])

  const LAST_JOB_KEY = 'sarabcraft.imageAttack.lastJobId'

  // Persist the current job (running OR completed) so a hard refresh can
  // re-attach to it — a running job will resume live progress, a completed one
  // restores its result, both with their input/target images.
  useEffect(() => {
    if (currentJobId && (loading || result)) {
      try { window.localStorage.setItem(LAST_JOB_KEY, currentJobId) } catch { /* ignore */ }
    }
  }, [result, loading, currentJobId])

  // Pull both the input and target image artifacts back as data-URLs.
  const fetchJobImages = async (detail) => {
    const findArtifact = (field) => (detail.artifacts || []).find(
      (a) => a.role === `input-${field}` || a.metadata_json?.field === field
    )
    const toDataUrl = async (artifact) => {
      if (!artifact) return null
      try { return await getJobArtifactDataUrl(detail.job_id, artifact.id) } catch { return null }
    }
    const [inputDataUrl, targetDataUrl] = await Promise.all([
      toDataUrl(findArtifact('input_file')),
      toDataUrl(findArtifact('target_file')),
    ])
    return { inputDataUrl, targetDataUrl }
  }

  // Re-attach the page to an existing job: restore form/images, and if the job
  // is still queued/running, resume live-progress polling until it completes.
  const attachToJob = async (jobId, { signal } = {}) => {
    const detail = await getJobDetails(jobId)
    if (detail?.kind !== 'image_attack') {
      throw new Error('Only image attacks can be opened on this page')
    }
    const isLive = ['queued', 'running'].includes(detail.status)
    if (!isLive && (detail.status !== 'completed' || !detail?.result?.adversarial_b64)) {
      // Nothing useful to restore (failed/cancelled with no result).
      try { window.localStorage.removeItem(LAST_JOB_KEY) } catch { /* ignore */ }
      throw new Error(detail.error_message || 'Job has no recoverable result')
    }
    const { inputDataUrl, targetDataUrl } = await fetchJobImages(detail)
    restoreFromJob(detail, { inputDataUrl, targetDataUrl, live: isLive })

    if (!isLive) return

    // Resume live progress on a running job. Detaching (navigation/refresh)
    // leaves the backend job running; we never cancel it here.
    try {
      const nextResult = await followImageAttackJob(jobId, {
        signal,
        onProgress: (p) => setProgress(p),
      })
      setResult(nextResult)
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message)
    } finally {
      setLoading(false)
      setProgress(null)
    }
  }

  // On mount: re-attach to a job from the ?job=<id> deep link (preferred) or
  // the last saved job in localStorage. Handles running AND completed jobs.
  useEffect(() => {
    const deepLinkJob = searchParams.get('job') || ''
    let savedJobId = deepLinkJob
    if (!savedJobId) {
      // Don't clobber an in-memory session that's already showing something.
      if (result || loading) return
      try { savedJobId = window.localStorage.getItem(LAST_JOB_KEY) || '' } catch { savedJobId = '' }
    }
    if (!savedJobId) return
    // Already attached to this exact job — nothing to do.
    if (savedJobId === currentJobId && (result || loading)) {
      if (deepLinkJob) { const sp = new URLSearchParams(searchParams); sp.delete('job'); setSearchParams(sp, { replace: true }) }
      return
    }

    const { signal, abort } = createAbortable()
    abortRef.current = { abort }
    ;(async () => {
      try {
        await attachToJob(savedJobId, { signal })
      } catch {
        /* job gone or not recoverable — leave the form as-is */
      } finally {
        // Clear the deep-link param so a later manual run doesn't re-trigger.
        if (deepLinkJob) {
          const sp = new URLSearchParams(searchParams)
          sp.delete('job')
          setSearchParams(sp, { replace: true })
        }
      }
    })()
    return () => abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  const attackMeta = registry[attack] || {}
  const attackParams = attackMeta.params || {}

  const getParam = (key) => {
    if (paramValues[key] !== undefined) return paramValues[key]
    return attackParams[key]?.default ?? 0
  }
  const setParam = (key, val) => setParamValues(p => ({ ...p, [key]: val }))

  const buildAttackPayload = () => buildAttackParamPayload(attack, attackParams, getParam)

  const handleAttackChange = (name) => {
    setAttack(name)
    setParamValues({})
    setInfoOpen(null)
  }

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

  const handleRun = async () => {
    if (!inputFile || !targetFile) { setError('Upload both input and target images'); return }
    abortRef.current?.abort()
    const { signal, abort } = createAbortable()
    abortRef.current = { abort }
    jobIdRef.current = ''
    setLoading(true)
    setError('')
    setResult(null)
    setCurrentJobId('')
    setProgress(null)
    try { window.localStorage.removeItem('sarabcraft.imageAttack.lastJobId') } catch { /* ignore */ }
    try {
      const fd = new FormData()
      fd.append('input_file', inputFile); fd.append('target_file', targetFile)
      fd.append('model', model); fd.append('attack', attack)
      const payloadParams = buildAttackPayload()
      Object.entries(payloadParams).forEach(([key, val]) => {
        if (key === 'epsilon') fd.append('epsilon', String(val))
        else if (key === 'iterations') fd.append('iterations', String(val))
        else fd.append(key, String(val))
      })
      if (!attackParams.epsilon) fd.append('epsilon', '16')
      if (!attackParams.iterations) fd.append('iterations', '50')
      fd.append('ensemble_mode', ensembleMode)
      if (ensembleModels.length) fd.append('ensemble_models', ensembleModels.join(','))
      const nextResult = await runImageAttack(fd, {
        signal,
        cancelOnAbort: false,
        onCreated: (job) => {
          jobIdRef.current = job.job_id
          setCurrentJobId(job.job_id)
          // Re-attach a progress poller to this job; runImageAttack itself only
          // resolves with the final result. This poller is read-only and never
          // cancels the job (it stops once the job leaves a live state).
          followImageAttackJob(job.job_id, {
            signal,
            onProgress: (p) => setProgress(p),
          }).catch(() => {})
        },
      })
      setResult(nextResult)
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message)
    } finally {
      setLoading(false)
      setProgress(null)
      abortRef.current = null
    }
  }

  const handleCancel = async () => {
    const jobId = jobIdRef.current || currentJobId
    abortRef.current?.abort()
    abortRef.current = null
    setLoading(false)
    setProgress(null)
    try { window.localStorage.removeItem('sarabcraft.imageAttack.lastJobId') } catch { /* ignore */ }
    setError('Cancellation requested. Open Jobs to monitor or resume later.')
    if (!jobId) return
    try {
      await cancelJobById(jobId)
    } catch (e) {
      setError(e.message || 'Failed to cancel job')
    }
  }

  const modelOpts = modelsList.length ? modelsList.map(item => ({ value: item.value, label: item.label })) : [{ value: 'microsoft/resnet-50', label: 'ResNet-50' }]

  return (
    <div className="space-y-4">
      {/* Row 1: Input + Configuration side by side (matching benchmark layout) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Input</SectionLabel>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <ImageUpload label="Input image" preview={inputPreview} onChange={pickFile(setInputFile, setInputPreview)} />
            <ImageUpload label="Target image" preview={targetPreview} onChange={pickFile(setTargetFile, setTargetPreview)} />
          </div>
          <ModelPicker label="Model" value={model} onChange={setModel} models={modelsList} />
        </Card>

        <Card>
          <SectionLabel>Attack Method</SectionLabel>
          <AttackSelector grouped={grouped} attack={attack} searchTerm={searchTerm} setSearchTerm={setSearchTerm} onSelect={handleAttackChange} onInfo={name => setInfoOpen(infoOpen === name ? null : name)} />
        </Card>
      </div>

      <AttackInfoModal name={infoOpen} meta={infoOpen ? registry[infoOpen] : null} onClose={() => setInfoOpen(null)} />

      {/* Row 2: Parameters + Ensemble */}
      {Object.keys(attackParams).length > 0 && (
        <Card>
          <SectionLabel>
            Parameters — {attack}
            <span className={`ml-2 text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[attackMeta.threat] || THREAT_COLORS.whitebox}`}>
              {THREAT_LABEL[attackMeta.threat] || 'White-box'}
            </span>
          </SectionLabel>
          <ParamRenderer attackName={attack} params={attackParams} getParam={getParam} setParam={setParam} />
          <EnsemblePanel attack={attack} modelOpts={modelOpts} model={model}
            ensembleModels={ensembleModels} setEnsembleModels={setEnsembleModels}
            ensembleMode={ensembleMode} setEnsembleMode={setEnsembleMode} />
        </Card>
      )}

      {/* Run */}
      <div className="flex items-center gap-4">
        <RunButton onClick={handleRun} loading={loading} label="Run Attack" />
        {loading && (
          <button onClick={handleCancel}
            className="px-4 py-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg text-sm hover:bg-red-500/20 transition font-medium">
            Cancel
          </button>
        )}
        <ErrorMsg msg={error} />
      </div>

      {(loading || currentJobId) && (
        <Card className="border-[var(--accent)]/20">
          <SectionLabel>Durable Job Status</SectionLabel>
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <p className="text-sm font-semibold text-slate-200">This attack is running as a background job.</p>
              <p className="text-xs text-slate-500 mt-1">
                {currentJobId ? `Job ID: ${currentJobId}` : 'Waiting for job creation...'}
              </p>
            </div>
            {currentJobId && (
              <Link
                to={`/jobs?job=${currentJobId}`}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 hover:bg-cyan-500/20 transition"
              >
                Open in Jobs
              </Link>
            )}
          </div>
          {loading && (
            <div className="mt-3 space-y-1.5">
              <div className="flex items-center justify-between text-[11px] text-slate-500">
                <span>{progress?.message || 'Waiting for worker...'}</span>
                <span>{progress?.current ?? 0}/{progress?.total ?? 1}</span>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-700/50 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-cyan-400 transition-all duration-300"
                  style={{ width: `${Math.max(0, Math.min(100, progress?.percent ?? 0))}%` }}
                />
              </div>
            </div>
          )}
          <p className="mt-3 text-[11px] text-slate-500">
            Jobs keeps queue state, cancellation, and resume controls visible even after refresh or navigation. This page auto-reconnects to a running job on refresh.
          </p>
        </Card>
      )}

      {/* Result images (shown after attack completes) */}
      {result && (
        <Card>
          <SectionLabel>Output Images</SectionLabel>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <ImageResult label="Input" src={inputPreview} />
            <ImageResult label="Target" src={targetPreview} />
            <ImageResult label="Adversarial" src={`data:image/png;base64,${result.adversarial_b64}`}
              onDownload={() => downloadB64(result.adversarial_b64, `adversarial_${slugify(attack)}.png`, 'image/png')} />
            {result.perturbation_b64 && <ImageResult label="Perturbation (10×)" src={`data:image/png;base64,${result.perturbation_b64}`}
              onDownload={() => downloadB64(result.perturbation_b64, `perturbation_${slugify(attack)}.png`, 'image/png')} />}
          </div>
        </Card>
      )}

      {/* Results */}
      {result && (
        <Card className="border-[var(--accent)]/30">
          <SectionLabel>Results</SectionLabel>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <p className={`text-lg font-bold mb-3 ${result.success ? 'text-green-400' : 'text-amber-400'}`}>{result.status}</p>
              <div className="text-xs text-slate-500 mb-4">
                {attack} | {modelsList.find(item => item.value === model)?.label || model} | {attackMeta.threat} | {attackMeta.norm}
                {ensembleModels.length ? ` | Ensemble: ${ensembleModels.length} models (${ensembleMode})` : ''}
              </div>
              <PredBars label="Original predictions" preds={result.original_preds} highlight={result.original_class} color="blue" />
              <PredBars label="Adversarial predictions" preds={result.adversarial_preds} highlight={result.target_class} color="green" />
            </div>
            <div className="flex flex-col items-center gap-4">
              <div className="flex flex-wrap gap-2 justify-center">
                {result.adversarial_b64 && (
                  <DownloadBtn label="Adversarial PNG" onClick={() => downloadB64(result.adversarial_b64, `adversarial_${slugify(attack)}.png`, 'image/png')} />
                )}
                {result.perturbation_b64 && (
                  <DownloadBtn label="Perturbation PNG" onClick={() => downloadB64(result.perturbation_b64, `perturbation_${slugify(attack)}.png`, 'image/png')} />
                )}
              </div>
              <button onClick={() => setTransferModal(true)}
                className="px-5 py-2.5 bg-[var(--accent-dim)] border border-[var(--accent)] text-[var(--accent)] font-semibold rounded-lg hover:bg-[var(--accent)]/20 transition text-sm">
                Test Transfer on External Models
              </button>
            </div>
          </div>

          {/* Perturbation Metrics */}
          {result.metrics && (
            <div className="mt-6 pt-6 border-t border-slate-700/50">
              <MetricsPanel metrics={result.metrics} />
            </div>
          )}

          {/* GradCAM */}
          {result.adversarial_b64 && inputPreview && (
            <div className="mt-4 pt-4 border-t border-slate-700/50">
              <GradCAMPanel
                originalB64={inputPreview?.split(',')[1]}
                adversarialB64={result.adversarial_b64}
                model={model}
              />
            </div>
          )}

          {/* Report Export */}
          <div className="mt-4 pt-4 border-t border-slate-700/50 flex flex-wrap gap-2">
            <button onClick={() => downloadReport(result, attack, model, result.metrics)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-700/60 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition text-[11px]">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>
              Export HTML Report
            </button>
            <button onClick={() => downloadJSON(result, attack, model, result.metrics)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-700/60 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition text-[11px]">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>
              Export JSON
            </button>
          </div>
        </Card>
      )}

      {transferModal && result && (
        <TransferModal adversarialB64={result.adversarial_b64} originalB64={inputPreview?.split(',')[1]} targetLabel={result.target_class} originalLabel={result.original_class} onClose={() => setTransferModal(false)} />
      )}
    </div>
  )
}

// ── Small private components ────────────────────────────────────────────────

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
            <div className="px-3 py-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-wider bg-slate-800/40 sticky top-0 z-10">{cat}</div>
            {attacks.map(a => {
              const sel = a.name === attack
              return (
                <div key={a.name} onClick={() => onSelect(a.name)}
                  className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${sel ? 'bg-[var(--accent)]/10 border-l-2 border-[var(--accent)]' : 'hover:bg-slate-800/50 border-l-2 border-transparent'}`}>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm truncate ${sel ? 'text-[var(--accent)] font-semibold' : 'text-slate-200'}`}>{a.name}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[a.threat] || THREAT_COLORS.whitebox}`}>{THREAT_LABEL[a.threat] || 'White-box'}</span>
                      <span className="text-[10px] text-slate-600">{a.year}</span>
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

const pickFile = (setter, previewSetter) => e => {
  const f = e.target.files?.[0]
  setter(f)
  if (f) { const r = new FileReader(); r.onload = ev => previewSetter(ev.target.result); r.readAsDataURL(f) }
  else previewSetter(null)
}

function ImageUpload({ label, preview, onChange }) {
  return (
    <div>
      <span className="block text-xs text-slate-400 mb-1">{label}</span>
      <label className="block aspect-square border-2 border-dashed border-slate-600 hover:border-[var(--accent)] rounded-lg cursor-pointer overflow-hidden bg-slate-800/60 transition-colors relative group">
        {preview
          ? <img src={preview} alt="" className="w-full h-full object-cover" />
          : <div className="flex items-center justify-center h-full text-slate-500 text-xs">Click to upload</div>
        }
        <input type="file" accept="image/*" onChange={onChange} className="absolute inset-0 opacity-0 cursor-pointer" />
      </label>
    </div>
  )
}

function ImageResult({ label, src, onDownload }) {
  return (
    <div>
      <span className="block text-xs text-slate-400 mb-1">{label}</span>
      <div className="aspect-square rounded-lg overflow-hidden bg-slate-800/60 border border-slate-600 relative group">
        <img src={src} alt={label} className="w-full h-full object-cover" />
        {onDownload && (
          <button onClick={onDownload}
            className="absolute bottom-2 right-2 w-7 h-7 rounded-md bg-slate-900/70 hover:bg-[var(--accent)]/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity border border-slate-600/50"
            title={`Download ${label}`}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5 text-slate-200">
              <path d="M10 3a.75.75 0 01.75.75v7.69l2.22-2.22a.75.75 0 011.06 1.06l-3.5 3.5a.75.75 0 01-1.06 0l-3.5-3.5a.75.75 0 111.06-1.06l2.22 2.22V3.75A.75.75 0 0110 3z" />
              <path d="M3 15.75a.75.75 0 01.75-.75h12.5a.75.75 0 010 1.5H3.75a.75.75 0 01-.75-.75z" />
            </svg>
          </button>
        )}
      </div>
    </div>
  )
}

function DownloadBtn({ label, onClick }) {
  return (
    <button onClick={onClick}
      className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-700/60 hover:bg-slate-600 text-slate-300 rounded-lg border border-slate-600 transition text-[11px]">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
        <path d="M10 3a.75.75 0 01.75.75v7.69l2.22-2.22a.75.75 0 011.06 1.06l-3.5 3.5a.75.75 0 01-1.06 0l-3.5-3.5a.75.75 0 111.06-1.06l2.22 2.22V3.75A.75.75 0 0110 3z" />
        <path d="M3 15.75a.75.75 0 01.75-.75h12.5a.75.75 0 010 1.5H3.75a.75.75 0 01-.75-.75z" />
      </svg>
      {label}
    </button>
  )
}

function PredBars({ label, preds, highlight, color }) {
  if (!preds) return null
  const c = color === 'green' ? 'bg-green-500/60' : 'bg-blue-500/60'
  const ch = color === 'green' ? 'bg-green-400' : 'bg-blue-400'
  return (
    <div className="mb-4">
      <p className="text-xs text-slate-500 mb-2">{label}</p>
      <div className="space-y-1.5">
        {Object.entries(preds).slice(0, 5).map(([k, v]) => (
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
