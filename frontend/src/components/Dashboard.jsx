import { useState, useEffect, useCallback } from 'react'
import { getHistoryStats, getHeatmapData, getHistory, clearHistory } from '../api/client'
import { Card, SectionLabel } from './ui/Section'
import { CardSkeleton, TableSkeleton } from './ui/Skeleton'

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [recent, setRecent] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(() => {
    setLoading(true)
    Promise.all([
      getHistoryStats(),
      getHeatmapData(),
      getHistory(20, 0),
    ]).then(([s, h, r]) => {
      setStats(s)
      setHeatmap(h)
      setRecent(r.entries || [])
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])

  useEffect(() => { load() }, [load])

  const handleClear = () => {
    if (!confirm('Clear all attack history? This cannot be undone.')) return
    clearHistory().then(load)
  }

  if (loading) return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({length: 4}, (_, i) => <div key={i} className="h-24 bg-slate-800/40 border border-slate-700/50 rounded-xl animate-pulse" />)}
      </div>
      <CardSkeleton />
      <TableSkeleton />
    </div>
  )
  if (!stats || stats.total === 0) return <EmptyDashboard />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-200">Analytics Dashboard</h2>
          <p className="text-xs text-slate-500">Attack performance metrics and transferability analysis</p>
        </div>
        <div className="flex gap-2">
          <button onClick={load} className="px-3 py-1.5 bg-slate-700/60 border border-slate-600 text-slate-400 rounded-lg text-xs hover:text-slate-200 transition">
            Refresh
          </button>
          <button onClick={handleClear} className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-xs hover:bg-red-500/20 transition">
            Clear History
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Attacks" value={stats.total} />
        <StatCard label="Success Rate" value={`${(stats.success_rate * 100).toFixed(1)}%`} color={stats.success_rate > 0.5 ? 'text-emerald-400' : 'text-amber-400'} />
        <StatCard label="Unique Attacks" value={Object.keys(stats.attacks || {}).length} />
        <StatCard label="Unique Models" value={Object.keys(stats.models || {}).length} />
      </div>

      {/* Attack Success Chart */}
      <Card>
        <SectionLabel>Attack Success Rates</SectionLabel>
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {Object.entries(stats.attacks || {})
            .sort((a, b) => (b[1].success / b[1].total) - (a[1].success / a[1].total))
            .map(([name, data]) => {
              const rate = data.total > 0 ? data.success / data.total : 0
              return (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 w-40 truncate shrink-0" title={name}>{name}</span>
                  <div className="flex-1 bg-slate-700/60 rounded-full h-4 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${rate > 0.7 ? 'bg-emerald-500/70' : rate > 0.3 ? 'bg-amber-500/70' : 'bg-red-500/70'}`}
                      style={{ width: `${rate * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 w-20 text-right shrink-0">
                    {data.success}/{data.total} ({(rate * 100).toFixed(0)}%)
                  </span>
                </div>
              )
            })}
        </div>
      </Card>

      {/* Model Vulnerability Chart */}
      <Card>
        <SectionLabel>Model Vulnerability</SectionLabel>
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {Object.entries(stats.models || {})
            .sort((a, b) => (b[1].success / b[1].total) - (a[1].success / a[1].total))
            .map(([name, data]) => {
              const rate = data.total > 0 ? data.success / data.total : 0
              const shortName = name.split('/').pop()
              return (
                <div key={name} className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 w-48 truncate shrink-0" title={name}>{shortName}</span>
                  <div className="flex-1 bg-slate-700/60 rounded-full h-4 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${rate > 0.7 ? 'bg-red-500/70' : rate > 0.3 ? 'bg-amber-500/70' : 'bg-emerald-500/70'}`}
                      style={{ width: `${rate * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 w-20 text-right shrink-0">
                    {data.success}/{data.total} ({(rate * 100).toFixed(0)}%)
                  </span>
                </div>
              )
            })}
        </div>
      </Card>

      {/* Heatmap */}
      {heatmap && heatmap.attacks?.length > 0 && (
        <Card>
          <SectionLabel>Transferability Heatmap — Attack vs Model Success Rate</SectionLabel>
          <Heatmap data={heatmap} />
        </Card>
      )}

      {/* Recent History */}
      <Card>
        <SectionLabel>Recent Attacks</SectionLabel>
        {recent.length === 0
          ? <p className="text-xs text-slate-600">No attacks recorded yet.</p>
          : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700/50">
                    <th className="text-left py-2 pr-4 font-medium">Time</th>
                    <th className="text-left py-2 pr-4 font-medium">Attack</th>
                    <th className="text-left py-2 pr-4 font-medium">Model</th>
                    <th className="text-left py-2 pr-4 font-medium">Result</th>
                    <th className="text-right py-2 font-medium">L2</th>
                    <th className="text-right py-2 font-medium">SSIM</th>
                    <th className="text-right py-2 font-medium">PSNR</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map(e => (
                    <tr key={e.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                      <td className="py-2 pr-4 text-slate-500">{new Date(e.timestamp * 1000).toLocaleString()}</td>
                      <td className="py-2 pr-4 text-slate-300">{e.attack}</td>
                      <td className="py-2 pr-4 text-slate-400 truncate max-w-[160px]" title={e.model}>{e.model?.split('/').pop()}</td>
                      <td className="py-2 pr-4">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${e.success ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/15 text-red-400 border border-red-500/20'}`}>
                          {e.success ? 'SUCCESS' : 'FAILED'}
                        </span>
                      </td>
                      <td className="py-2 text-right text-slate-400">{e.metrics?.l2?.toFixed(4) ?? '—'}</td>
                      <td className="py-2 text-right text-slate-400">{e.metrics?.ssim?.toFixed(3) ?? '—'}</td>
                      <td className="py-2 text-right text-slate-400">{e.metrics?.psnr?.toFixed(1) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </Card>
    </div>
  )
}


function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-5 text-center">
      <div className={`text-2xl font-black ${color}`}>{value}</div>
      <div className="text-[10px] text-slate-500 mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}


function Heatmap({ data }) {
  const { attacks, models, matrix } = data
  if (!attacks.length || !models.length) return null

  const shortModels = models.map(m => m.split('/').pop()?.slice(0, 12) || m)

  return (
    <div className="overflow-x-auto">
      <div className="inline-block min-w-full">
        {/* Column headers */}
        <div className="flex">
          <div className="w-36 shrink-0" />
          {shortModels.map((m, i) => (
            <div key={i} className="w-14 shrink-0 text-center">
              <span className="text-[8px] text-slate-600 writing-mode-vertical transform -rotate-45 inline-block origin-bottom-left whitespace-nowrap" title={models[i]}>
                {m}
              </span>
            </div>
          ))}
        </div>
        {/* Rows */}
        {attacks.map((a, ri) => (
          <div key={a} className="flex items-center">
            <div className="w-36 shrink-0 text-[10px] text-slate-500 pr-2 truncate" title={a}>{a}</div>
            {matrix[ri].map((val, ci) => (
              <div key={ci} className="w-14 h-8 shrink-0 p-0.5">
                {val !== null ? (
                  <div
                    className="w-full h-full rounded-sm flex items-center justify-center text-[8px] font-bold"
                    style={{
                      backgroundColor: val > 0.7 ? `rgba(239,68,68,${0.2 + val * 0.6})` :
                                       val > 0.3 ? `rgba(251,191,36,${0.2 + val * 0.4})` :
                                                    `rgba(52,211,153,${0.3 + (1 - val) * 0.3})`,
                      color: val > 0.5 ? '#fff' : '#94a3b8'
                    }}
                    title={`${a} vs ${models[ci]}: ${(val * 100).toFixed(0)}%`}
                  >
                    {(val * 100).toFixed(0)}
                  </div>
                ) : (
                  <div className="w-full h-full rounded-sm bg-slate-800/40 flex items-center justify-center text-[8px] text-slate-700">—</div>
                )}
              </div>
            ))}
          </div>
        ))}
        {/* Legend */}
        <div className="flex items-center gap-3 mt-4 pl-36">
          <span className="text-[10px] text-slate-600">Success Rate:</span>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded-sm bg-emerald-500/40" /><span className="text-[9px] text-slate-600">Low</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded-sm bg-amber-500/40" /><span className="text-[9px] text-slate-600">Medium</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded-sm bg-red-500/60" /><span className="text-[9px] text-slate-600">High</span>
          </div>
        </div>
      </div>
    </div>
  )
}


function EmptyDashboard() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/50 flex items-center justify-center mb-6">
        <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
        </svg>
      </div>
      <h3 className="text-sm font-semibold text-slate-400 mb-2">No Data Yet</h3>
      <p className="text-xs text-slate-600 max-w-xs text-center">
        Run some attacks to populate the dashboard. Every image and audio attack is automatically tracked and analyzed.
      </p>
    </div>
  )
}
