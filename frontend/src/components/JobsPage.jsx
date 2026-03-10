import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { cancelJobById, getJobDetails, getJobs, resumeJobById } from '../api/client'
import { Card, ErrorMsg, SectionLabel, Select } from './ui/Section'
import { useConfirm } from './ui/ConfirmDialog'
import { useToast } from './ui/Toast'

const STATUS_OPTIONS = ['All statuses', 'queued', 'running', 'completed', 'failed', 'cancelled']

const STATUS_STYLES = {
  queued: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
  running: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
  completed: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  failed: 'bg-red-500/15 text-red-300 border-red-500/30',
  cancelled: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
}

function shortId(id = '') {
  return id ? `${id.slice(0, 8)}...${id.slice(-4)}` : '-'
}

function formatDate(value) {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '-' : date.toLocaleString()
}

function canCancel(job) {
  return ['queued', 'running'].includes(job?.status)
}

function canResume(job) {
  return Boolean(job?.resume_supported && ['failed', 'cancelled'].includes(job?.status))
}

function sanitizeForDisplay(value, key = '') {
  if (Array.isArray(value)) return value.map((item) => sanitizeForDisplay(item, key))
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([childKey, childValue]) => [childKey, sanitizeForDisplay(childValue, childKey)])
    )
  }
  if (typeof value === 'string') {
    if (key.endsWith('_b64')) return `[${key} omitted: ${value.length} chars]`
    if (value.length > 180) return `${value.slice(0, 140)}... (${value.length} chars)`
  }
  return value
}

function summaryLines(job) {
  const result = job?.result || {}
  const lines = []
  if (typeof result.success === 'boolean') lines.push(result.success ? 'Marked successful' : 'Marked unsuccessful')
  if (typeof result.success_rate === 'number') lines.push(`Success rate ${(result.success_rate * 100).toFixed(1)}%`)
  if (typeof result.attack_success_rate === 'number') lines.push(`Attack success rate ${(result.attack_success_rate * 100).toFixed(1)}%`)
  if (typeof result.transfer_rate === 'number') lines.push(`Transfer rate ${(result.transfer_rate * 100).toFixed(1)}%`)
  if (typeof result.total === 'number') lines.push(`${result.total} total items`)
  if (typeof result.total_models === 'number') lines.push(`${result.total_models} models tested`)
  if (Array.isArray(result.results)) lines.push(`${result.results.length} streamed result rows`)
  return lines
}

function StatusBadge({ status }) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide border ${STATUS_STYLES[status] || STATUS_STYLES.queued}`}>
      {status}
    </span>
  )
}

function ProgressBar({ progress }) {
  const percent = Math.max(0, Math.min(100, progress?.percent ?? 0))
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px] text-slate-500">
        <span>{progress?.message || 'Waiting for worker'}</span>
        <span>{progress?.current ?? 0}/{progress?.total ?? 1}</span>
      </div>
      <div className="w-full h-2 rounded-full bg-slate-700/50 overflow-hidden">
        <div className="h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-cyan-400 transition-all duration-300" style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}

function StatCard({ label, value, accent = 'text-slate-200' }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3">
      <div className={`text-sm font-semibold ${accent}`}>{value}</div>
      <div className="text-[10px] text-slate-500 uppercase tracking-widest mt-1">{label}</div>
    </div>
  )
}

export default function JobsPage() {
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [statusFilter, setStatusFilter] = useState('All statuses')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyJobId, setBusyJobId] = useState('')
  const [searchParams, setSearchParams] = useSearchParams()
  const { confirm } = useConfirm()
  const { toast } = useToast()

  const selectedJobId = searchParams.get('job') || ''
  const statusValue = statusFilter === 'All statuses' ? '' : statusFilter

  const loadJobs = useCallback(async () => {
    try {
      const data = await getJobs(75, statusValue)
      setJobs(data.jobs || [])
      setError('')
    } catch (err) {
      setError(err.message || 'Failed to load jobs')
    } finally {
      setLoading(false)
    }
  }, [statusValue])

  const loadSelectedJob = useCallback(async (jobId = selectedJobId) => {
    if (!jobId) {
      setSelectedJob(null)
      return
    }
    try {
      const detail = await getJobDetails(jobId)
      setSelectedJob(detail)
      setError('')
    } catch (err) {
      setError(err.message || 'Failed to load job details')
    }
  }, [selectedJobId])

  useEffect(() => {
    loadJobs()
    const timer = window.setInterval(loadJobs, 2000)
    return () => window.clearInterval(timer)
  }, [loadJobs])

  useEffect(() => {
    loadSelectedJob()
    if (!selectedJobId) return undefined
    const timer = window.setInterval(() => loadSelectedJob(selectedJobId), 2000)
    return () => window.clearInterval(timer)
  }, [loadSelectedJob, selectedJobId])

  useEffect(() => {
    if (jobs.length === 0) {
      if (selectedJobId) setSearchParams({}, { replace: true })
      return
    }
    if (!selectedJobId || !jobs.some((job) => job.job_id === selectedJobId)) {
      setSearchParams({ job: jobs[0].job_id }, { replace: true })
    }
  }, [jobs, selectedJobId, setSearchParams])

  const counts = useMemo(() => {
    return jobs.reduce((acc, job) => {
      acc.total += 1
      acc[job.status] = (acc[job.status] || 0) + 1
      return acc
    }, { total: 0, queued: 0, running: 0, completed: 0, failed: 0, cancelled: 0 })
  }, [jobs])

  const handleSelect = (jobId) => {
    setSearchParams({ job: jobId }, { replace: true })
  }

  const handleCancel = async (job) => {
    const allowed = await confirm({
      title: 'Cancel Job?',
      message: `Cancel ${job.title} (${shortId(job.job_id)}) now?`,
      confirmText: 'Cancel job',
      variant: 'warning',
    })
    if (!allowed) return
    setBusyJobId(job.job_id)
    try {
      await cancelJobById(job.job_id)
      toast({ type: 'info', message: `Cancellation requested for ${shortId(job.job_id)}.` })
      await Promise.all([loadJobs(), loadSelectedJob(job.job_id)])
    } catch (err) {
      setError(err.message || 'Failed to cancel job')
    } finally {
      setBusyJobId('')
    }
  }

  const handleResume = async (job) => {
    const allowed = await confirm({
      title: 'Resume Job?',
      message: `Resume ${job.title} (${shortId(job.job_id)}) from its saved checkpoint?`,
      confirmText: 'Resume job',
    })
    if (!allowed) return
    setBusyJobId(job.job_id)
    try {
      await resumeJobById(job.job_id)
      toast({ type: 'success', message: `Resumed ${shortId(job.job_id)}.` })
      await Promise.all([loadJobs(), loadSelectedJob(job.job_id)])
    } catch (err) {
      setError(err.message || 'Failed to resume job')
    } finally {
      setBusyJobId('')
    }
  }

  const detailSummary = summaryLines(selectedJob)
  const sanitizedRequest = selectedJob ? JSON.stringify(sanitizeForDisplay(selectedJob.request), null, 2) : ''
  const sanitizedResult = selectedJob?.result ? JSON.stringify(sanitizeForDisplay(selectedJob.result), null, 2) : ''
  const sanitizedCheckpoint = selectedJob?.checkpoint ? JSON.stringify(sanitizeForDisplay(selectedJob.checkpoint), null, 2) : ''
  const recentEvents = [...(selectedJob?.events || [])].reverse().slice(0, 12)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-200">Jobs Queue</h2>
          <p className="text-xs text-slate-500">Monitor queued work, cancel running jobs, and resume checkpointed jobs.</p>
        </div>
        <div className="flex items-center gap-3">
          <Select
            label="Status"
            value={statusFilter}
            onChange={setStatusFilter}
            options={STATUS_OPTIONS}
            className="min-w-[180px]"
          />
          <button
            onClick={loadJobs}
            className="px-3 py-2 bg-slate-700/60 border border-slate-600 text-slate-300 rounded-lg text-xs hover:bg-slate-600 transition mt-5"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard label="Visible Jobs" value={counts.total} />
        <StatCard label="Queued" value={counts.queued} accent="text-slate-300" />
        <StatCard label="Running" value={counts.running} accent="text-cyan-300" />
        <StatCard label="Completed" value={counts.completed} accent="text-emerald-300" />
        <StatCard label="Failed / Cancelled" value={counts.failed + counts.cancelled} accent="text-amber-300" />
      </div>

      <ErrorMsg msg={error} />

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.25fr)_minmax(360px,0.95fr)] gap-4">
        <Card>
          <SectionLabel>Recent Jobs</SectionLabel>
          {loading ? (
            <div className="text-sm text-slate-500">Loading jobs...</div>
          ) : jobs.length === 0 ? (
            <div className="text-sm text-slate-500">No jobs match the current filter.</div>
          ) : (
            <div className="space-y-2">
              {jobs.map((job) => {
                const active = job.job_id === selectedJobId
                const busy = busyJobId === job.job_id
                return (
                  <div
                    key={job.job_id}
                    className={`w-full text-left rounded-xl border p-4 transition ${active ? 'border-[var(--accent)] bg-[var(--accent)]/8' : 'border-slate-700/60 bg-slate-800/30 hover:bg-slate-800/50'}`}
                  >
                    <div className="flex items-start gap-3">
                      <button onClick={() => handleSelect(job.job_id)} className="min-w-0 flex-1 text-left">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <p className="text-sm font-semibold text-slate-200 truncate">{job.title}</p>
                            <StatusBadge status={job.status} />
                          </div>
                          <p className="text-[11px] text-slate-500 mt-1">{job.kind} · {shortId(job.job_id)} · {formatDate(job.created_at)}</p>
                        </div>
                        <div className="mt-3">
                          <ProgressBar progress={job.progress} />
                        </div>
                      </button>
                      <div className="flex items-center gap-2 shrink-0">
                        {canCancel(job) && (
                          <button
                            onClick={() => handleCancel(job)}
                            disabled={busy}
                            className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border ${busy ? 'opacity-50' : 'hover:bg-red-500/20'} bg-red-500/10 border-red-500/20 text-red-400 transition`}
                          >
                            Cancel
                          </button>
                        )}
                        {canResume(job) && (
                          <button
                            onClick={() => handleResume(job)}
                            disabled={busy}
                            className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border ${busy ? 'opacity-50' : 'hover:bg-cyan-500/20'} bg-cyan-500/10 border-cyan-500/20 text-cyan-300 transition`}
                          >
                            Resume
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </Card>

        <Card>
          <SectionLabel>Job Detail</SectionLabel>
          {!selectedJob ? (
            <div className="text-sm text-slate-500">Select a job to inspect progress, events, and controls.</div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="text-base font-semibold text-slate-200">{selectedJob.title}</h3>
                    <StatusBadge status={selectedJob.status} />
                  </div>
                  <p className="text-[11px] text-slate-500 mt-1">{selectedJob.kind} · {selectedJob.domain} · {selectedJob.job_id}</p>
                </div>
                <div className="flex items-center gap-2">
                  {canCancel(selectedJob) && (
                    <button
                      onClick={() => handleCancel(selectedJob)}
                      disabled={busyJobId === selectedJob.job_id}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 disabled:opacity-40 transition"
                    >
                      Cancel
                    </button>
                  )}
                  {canResume(selectedJob) && (
                    <button
                      onClick={() => handleResume(selectedJob)}
                      disabled={busyJobId === selectedJob.job_id}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 hover:bg-cyan-500/20 disabled:opacity-40 transition"
                    >
                      Resume
                    </button>
                  )}
                </div>
              </div>

              <ProgressBar progress={selectedJob.progress} />

              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Created" value={formatDate(selectedJob.created_at)} />
                <StatCard label="Updated" value={formatDate(selectedJob.updated_at)} />
                <StatCard label="Attempts" value={selectedJob.attempts ?? 0} />
                <StatCard label="Artifacts" value={selectedJob.artifacts?.length ?? 0} />
              </div>

              {detailSummary.length > 0 && (
                <div className="text-xs text-slate-400 space-y-1">
                  {detailSummary.map((line) => (
                    <p key={line}>{line}</p>
                  ))}
                </div>
              )}

              {selectedJob.error_message && (
                <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                  {selectedJob.error_message}
                </div>
              )}

              {selectedJob.artifacts?.length > 0 && (
                <div className="space-y-2">
                  <SectionLabel>Artifacts</SectionLabel>
                  <div className="space-y-1.5">
                    {selectedJob.artifacts.map((artifact) => (
                      <div key={artifact.id} className="flex items-center justify-between gap-2 text-[11px] text-slate-400 bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-2">
                        <span className="truncate">{artifact.role} · {artifact.filename || artifact.storage_key}</span>
                        <span className="shrink-0">{artifact.size_bytes} B</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="space-y-2">
                <SectionLabel>Recent Events</SectionLabel>
                {recentEvents.length === 0 ? (
                  <div className="text-xs text-slate-500">No events recorded yet.</div>
                ) : (
                  <div className="space-y-1.5">
                    {recentEvents.map((event) => (
                      <div key={event.id} className="bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-2">
                        <div className="flex items-center justify-between gap-3 text-[11px]">
                          <span className="font-semibold uppercase tracking-wide text-slate-300">{event.event_type}</span>
                          <span className="text-slate-500">{formatDate(event.created_at)}</span>
                        </div>
                        {Object.keys(event.payload_json || {}).length > 0 && (
                          <pre className="mt-1.5 text-[11px] text-slate-400 whitespace-pre-wrap break-all">
                            {JSON.stringify(sanitizeForDisplay(event.payload_json), null, 2)}
                          </pre>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <div>
                  <SectionLabel>Request</SectionLabel>
                  <pre className="text-[11px] text-slate-400 bg-slate-900/80 border border-slate-700/60 rounded-lg p-3 overflow-auto max-h-56 whitespace-pre-wrap break-all">
                    {sanitizedRequest}
                  </pre>
                </div>
                {selectedJob.result && (
                  <div>
                    <SectionLabel>Result</SectionLabel>
                    <pre className="text-[11px] text-slate-400 bg-slate-900/80 border border-slate-700/60 rounded-lg p-3 overflow-auto max-h-56 whitespace-pre-wrap break-all">
                      {sanitizedResult}
                    </pre>
                  </div>
                )}
                {selectedJob.checkpoint && Object.keys(selectedJob.checkpoint).length > 0 && (
                  <div>
                    <SectionLabel>Checkpoint</SectionLabel>
                    <pre className="text-[11px] text-slate-400 bg-slate-900/80 border border-slate-700/60 rounded-lg p-3 overflow-auto max-h-40 whitespace-pre-wrap break-all">
                      {sanitizedCheckpoint}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
