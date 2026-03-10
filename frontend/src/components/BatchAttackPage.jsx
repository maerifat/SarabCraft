import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { runBatchAttackStream, getModels, getAttacks } from '../api/client'
import { Card, SectionLabel, Select, RunButton, ErrorMsg } from './ui/Section'
import ModelPicker from './ui/ModelPicker'
import { SARABCRAFT_R1_NAME } from './image/sarabcraftR1'

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

export default function BatchAttackPage() {
  const [files, setFiles] = useState([])
  const [targetFile, setTargetFile] = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [model, setModel] = useState('')
  const [attack, setAttack] = useState('PGD')
  const [epsilon, setEpsilon] = useState(16)
  const [iterations, setIterations] = useState(40)
  const [modelsList, setModelsList] = useState([])
  const [attackList, setAttackList] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [jobId, setJobId] = useState('')
  const [progress, setProgress] = useState({ current: 0, total: 0, message: '' })
  const [currentFile, setCurrentFile] = useState('')
  const abortRef = useRef(null)

  useEffect(() => {
    Promise.all([getModels(), getAttacks()]).then(([m, a]) => {
      const items = normalizeModelItems(m)
      setModelsList(items)
      if (items.length > 0) setModel(prev => prev || m.default_id || items[0].value)
      setAttackList(Object.keys(a.registry || {}))
    }).catch(() => {})
  }, [])

  useEffect(() => () => {
    abortRef.current = null
  }, [])

  const handleFiles = e => {
    const arr = Array.from(e.target.files || [])
    setFiles(arr)
  }

  const handleTarget = e => {
    const f = e.target.files?.[0]
    setTargetFile(f)
    if (f) { const r = new FileReader(); r.onload = ev => setTargetPreview(ev.target.result); r.readAsDataURL(f) }
  }

  const handleRun = () => {
    if (!files.length || !targetFile) { setError('Upload input images and a target image'); return }
    setLoading(true)
    setError('')
    setResult(null)
    setJobId('')
    setCurrentFile('')
    setProgress({ current: 0, total: files.length, message: 'Submitting batch job' })

    const fd = new FormData()
    files.forEach(f => fd.append('input_files', f))
    fd.append('target_file', targetFile)
    fd.append('model', model)
    fd.append('attack', attack)
    fd.append('epsilon', String(epsilon))
    fd.append('iterations', String(iterations))

    const handle = runBatchAttackStream(fd, {
      onCreated: (job) => {
        setJobId(job.job_id)
        if (job.progress) setProgress(job.progress)
      },
      onInit: (data) => {
        setProgress({
          current: data.resume_from || 0,
          total: data.total || files.length,
          message: `Queued ${data.total || files.length} images`,
        })
      },
      onProgress: (data) => {
        setCurrentFile(data.filename || '')
        setProgress(prev => ({
          current: data.index ?? prev.current,
          total: data.total || prev.total || files.length,
          message: data.filename ? `Running ${data.filename}` : 'Running batch attack',
        }))
      },
      onResult: (data) => {
        setCurrentFile(data.filename || '')
        setProgress(prev => ({
          current: typeof data.index === 'number' ? data.index + 1 : prev.current + 1,
          total: prev.total || files.length,
          message: data.filename ? `Finished ${data.filename}` : 'Received batch result',
        }))
      },
      onSummary: (data) => {
        setResult(data)
      },
      onDone: () => {
        setLoading(false)
        setCurrentFile('')
        abortRef.current = null
      },
      onError: (err) => {
        setError(err.message === 'Job cancelled' ? 'Job cancelled. Open Jobs to resume later.' : err.message)
        setLoading(false)
        setCurrentFile('')
        abortRef.current = null
      },
    })

    abortRef.current = handle
  }

  const handleAbort = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setLoading(false)
    setCurrentFile('')
    setError('Cancellation requested. Open Jobs to monitor or resume later.')
  }

  const canRun = files.length > 0 && targetFile
  const progressTotal = progress.total || files.length || 1
  const progressPct = Math.max(0, Math.min(100, Math.round((progress.current / progressTotal) * 100)))

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold text-slate-200">Batch Attack</h2>
        <p className="text-xs text-slate-500">Run the same attack on multiple images and get aggregate success rates</p>
      </div>

      <Card>
        <SectionLabel>Input Images</SectionLabel>
        <label className="block border-2 border-dashed border-slate-600 hover:border-[var(--accent)]/50 rounded-xl p-6 text-center cursor-pointer transition">
          <input type="file" accept="image/*" multiple onChange={handleFiles} className="hidden" />
          {files.length > 0 ? (
            <div>
              <p className="text-sm text-slate-300 font-medium">{files.length} images selected</p>
              <p className="text-xs text-slate-500 mt-1">{files.map(f => f.name).slice(0, 5).join(', ')}{files.length > 5 ? ` +${files.length - 5} more` : ''}</p>
            </div>
          ) : (
            <div>
              <svg className="w-8 h-8 mx-auto text-slate-600 mb-2" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
              </svg>
              <p className="text-sm text-slate-400">Click to select multiple images</p>
              <p className="text-xs text-slate-600 mt-1">Or drag and drop a folder of images</p>
            </div>
          )}
        </label>
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Target Image</SectionLabel>
          <div className="max-w-[180px] mb-3">
            <label className="block aspect-square border-2 border-dashed border-slate-600 hover:border-[var(--accent)]/50 rounded-lg cursor-pointer overflow-hidden bg-slate-800/60 transition relative">
              {targetPreview
                ? <img src={targetPreview} alt="" className="w-full h-full object-cover" />
                : <div className="flex items-center justify-center h-full text-slate-500 text-xs">Click to upload</div>
              }
              <input type="file" accept="image/*" onChange={handleTarget} className="absolute inset-0 opacity-0 cursor-pointer" />
            </label>
          </div>
        </Card>

        <Card>
          <SectionLabel>Configuration</SectionLabel>
          <div className="space-y-3">
            <ModelPicker label="Model" value={model} onChange={setModel} models={modelsList} />
            <Select label="Attack" value={attack} onChange={setAttack} options={attackList.length ? attackList : ['PGD']} className="" />
            {attack === SARABCRAFT_R1_NAME && (
              <p className="text-[11px] text-slate-500">
                Batch mode runs the standard R1 configuration. Use Image Attack or Robustness to enable multi-image transfer mode.
              </p>
            )}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <span className="text-xs text-slate-400 mb-1 block">Epsilon /255</span>
                <input type="number" value={epsilon} onChange={e => setEpsilon(+e.target.value)} min={1} max={255}
                  className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:border-[var(--accent)] outline-none" />
              </div>
              <div>
                <span className="text-xs text-slate-400 mb-1 block">Iterations</span>
                <input type="number" value={iterations} onChange={e => setIterations(+e.target.value)} min={1} max={500}
                  className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:border-[var(--accent)] outline-none" />
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div className="flex items-center gap-4">
        <RunButton onClick={handleRun} loading={loading} disabled={!canRun && !loading}
          label={`Run Batch (${files.length} images)`} loadingLabel="Processing batch..." />
        <ErrorMsg msg={error} />
      </div>

      {(loading || jobId) && (
        <Card className="border-[var(--accent)]/20">
          <SectionLabel>Durable Job Status</SectionLabel>
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <p className="text-sm font-semibold text-slate-200">Batch job is running through the persistent queue.</p>
              <p className="text-xs text-slate-500 mt-1">
                {jobId ? `Job ID: ${jobId}` : 'Waiting for job creation...'}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {jobId && (
                <Link
                  to={`/jobs?job=${jobId}`}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 hover:bg-cyan-500/20 transition"
                >
                  Open in Jobs
                </Link>
              )}
              {loading && (
                <button
                  onClick={handleAbort}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>

          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>{progress.message || 'Waiting for worker'}</span>
              <span>{progress.current}/{progressTotal}</span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-700/50 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-cyan-400 transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            {currentFile && (
              <p className="text-[11px] text-slate-500">
                Current file: <span className="text-slate-300">{currentFile}</span>
              </p>
            )}
            {!loading && jobId && !result && (
              <p className="text-[11px] text-slate-500">
                This job can be reopened from the Jobs page for status, cancellation, or resume.
              </p>
            )}
          </div>
        </Card>
      )}

      {result && <BatchResults data={result} />}
    </div>
  )
}


function BatchResults({ data }) {
  return (
    <Card className="border-[var(--accent)]/20">
      <SectionLabel>Batch Results — {data.attack} on {data.model?.split('/').pop()}</SectionLabel>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '1rem' }} className="mb-6">
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 text-center">
          <div className="text-2xl font-black text-white">{data.total}</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Total Images</div>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 text-center">
          <div className="text-2xl font-black text-emerald-400">{data.successes}</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Successful</div>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 text-center">
          <div className={`text-2xl font-black ${data.success_rate > 0.7 ? 'text-emerald-400' : data.success_rate > 0.3 ? 'text-amber-400' : 'text-red-400'}`}>
            {(data.success_rate * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Success Rate</div>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 text-center">
          <div className="text-2xl font-black text-slate-300">{data.avg_ssim?.toFixed(3)}</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Avg SSIM</div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-700/50 text-[10px] uppercase tracking-wider">
              <th className="text-left py-2 pr-3">#</th>
              <th className="text-left py-2 pr-3">File</th>
              <th className="text-left py-2 pr-3">Original</th>
              <th className="text-left py-2 pr-3">Adversarial</th>
              <th className="text-left py-2 pr-3">Result</th>
              <th className="text-right py-2 pr-3">L2</th>
              <th className="text-right py-2">SSIM</th>
            </tr>
          </thead>
          <tbody>
            {data.results?.map((r, i) => (
              <tr key={i} className="border-b border-slate-800/50">
                <td className="py-1.5 pr-3 text-slate-600">{i + 1}</td>
                <td className="py-1.5 pr-3 text-slate-400 truncate max-w-[140px]">{r.filename || '—'}</td>
                <td className="py-1.5 pr-3 text-slate-400 truncate max-w-[100px]">{r.original_class || '—'}</td>
                <td className="py-1.5 pr-3 text-slate-300 truncate max-w-[100px]">{r.adversarial_class || r.error?.slice(0, 30) || '—'}</td>
                <td className="py-1.5 pr-3">
                  {r.error ? (
                    <span className="text-red-400 text-[10px]">ERROR</span>
                  ) : (
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${r.success ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
                      {r.success ? 'HIT' : 'MISS'}
                    </span>
                  )}
                </td>
                <td className="py-1.5 pr-3 text-right text-slate-500">{r.metrics?.l2?.toFixed(4) ?? '—'}</td>
                <td className="py-1.5 text-right text-slate-500">{r.metrics?.ssim?.toFixed(3) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
