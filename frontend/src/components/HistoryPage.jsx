import { useState, useEffect, useCallback } from 'react'
import { getHistory, getHistoryEntry, deleteHistoryEntry, clearHistory } from '../api/client'
import { Card, SectionLabel } from './ui/Section'
import { downloadB64 } from '../utils/download'
import { TableSkeleton } from './ui/Skeleton'

export default function HistoryPage() {
  const [entries, setEntries] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [domain, setDomain] = useState('')
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)
  const [compareIds, setCompareIds] = useState([])
  const [compareData, setCompareData] = useState([])
  const LIMIT = 25

  const load = useCallback((p, d) => {
    setLoading(true)
    getHistory(LIMIT, p * LIMIT, d).then(r => {
      setEntries(r.entries || [])
      setTotal(r.total || 0)
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])

  useEffect(() => { load(page, domain) }, [load, page, domain])

  const handleDelete = (id) => {
    if (!confirm('Delete this history entry?')) return
    deleteHistoryEntry(id).then(() => load(page, domain))
  }

  const handleClear = () => {
    if (!confirm('Clear all attack history? This cannot be undone.')) return
    clearHistory().then(() => { setEntries([]); setTotal(0) })
  }

  const handleView = async (id) => {
    try {
      const full = await getHistoryEntry(id)
      setSelected(full)
    } catch (e) {
      console.error('Failed to load history entry:', e)
    }
  }

  const toggleCompare = (id) => {
    setCompareIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 4 ? [...prev, id] : prev
    )
  }

  const runCompare = async () => {
    try {
      const data = await Promise.all(compareIds.map(id => getHistoryEntry(id)))
      setCompareData(data)
    } catch (e) {
      console.error('Comparison failed:', e)
    }
  }

  const pages = Math.ceil(total / LIMIT)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-200">Attack History</h2>
          <p className="text-xs text-slate-500">{total} recorded attacks</p>
        </div>
        <div className="flex items-center gap-2">
          {/* Domain filter */}
          <select value={domain} onChange={e => { setDomain(e.target.value); setPage(0) }}
            className="bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-xs text-slate-200">
            <option value="">All domains</option>
            <option value="image">Image</option>
            <option value="audio">Audio</option>
          </select>
          {compareIds.length >= 2 && (
            <button onClick={runCompare}
              className="px-3 py-1.5 bg-[var(--accent)]/15 border border-[var(--accent)]/30 text-[var(--accent)] rounded-lg text-xs hover:bg-[var(--accent)]/25 transition">
              Compare ({compareIds.length})
            </button>
          )}
          <button onClick={handleClear}
            className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-xs hover:bg-red-500/20 transition">
            Clear All
          </button>
          <button onClick={() => window.open('/api/history/export?format=csv')}
            className="px-3 py-1.5 bg-slate-700/60 border border-slate-600 text-slate-300 rounded-lg text-xs hover:bg-slate-600 transition">
            Export CSV
          </button>
          <button onClick={() => {
            fetch('/api/history/export?format=json')
              .then(r => r.json())
              .then(data => {
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url; a.download = 'sarabcraft_history.json'; a.click()
                URL.revokeObjectURL(url)
              })
          }}
            className="px-3 py-1.5 bg-slate-700/60 border border-slate-600 text-slate-300 rounded-lg text-xs hover:bg-slate-600 transition">
            Export JSON
          </button>
        </div>
      </div>

      {/* Compare View */}
      {compareData.length >= 2 && (
        <Card className="border-[var(--accent)]/30">
          <SectionLabel>Comparison — {compareData.length} Attacks</SectionLabel>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700/50">
                  <th className="text-left py-2 pr-3 font-medium">Metric</th>
                  {compareData.map((e, i) => (
                    <th key={i} className="text-right py-2 px-2 font-medium text-slate-300">{e.attack}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const isAudio = compareData.some(e => e.domain === 'audio')
                  const metrics = isAudio
                    ? ['success', 'snr_db', 'wer']
                    : ['success', 'l2', 'linf_255', 'ssim', 'psnr']
                  return metrics.map(metric => (
                    <tr key={metric} className="border-b border-slate-800/50">
                      <td className="py-2 pr-3 text-slate-400 uppercase font-medium">{metric}</td>
                      {compareData.map((e, i) => {
                        const val = metric === 'success' ? (e.success ? 'YES' : 'NO') : (e.metrics?.[metric]?.toFixed?.(3) ?? e[metric]?.toFixed?.(3) ?? '—')
                        return <td key={i} className={`py-2 px-2 text-right ${metric === 'success' ? (e.success ? 'text-emerald-400' : 'text-red-400') : 'text-slate-300'}`}>{val}</td>
                      })}
                    </tr>
                  ))
                })()}
                <tr>
                  <td className="py-2 pr-3 text-slate-400 font-medium">PREVIEW</td>
                  {compareData.map((e, i) => (
                    <td key={i} className="py-2 px-2 text-right">
                      {e.adversarial_b64 && <img src={`data:image/png;base64,${e.adversarial_b64}`} alt="" className="w-14 h-14 object-cover rounded ml-auto border border-slate-700" />}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <button onClick={() => setCompareData([])} className="mt-3 text-xs text-slate-500 hover:text-slate-300">Close comparison</button>
        </Card>
      )}

      {/* Table */}
      {loading ? (
        <TableSkeleton rows={10} cols={8} />
      ) : entries.length === 0 ? (
        <EmptyState />
      ) : (
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700/50">
                  <th className="text-left py-2 pr-2 w-8">
                    <span className="text-[10px]">CMP</span>
                  </th>
                  <th className="text-left py-2 pr-4 font-medium">Time</th>
                  <th className="text-left py-2 pr-4 font-medium">Domain</th>
                  <th className="text-left py-2 pr-4 font-medium">Attack</th>
                  <th className="text-left py-2 pr-4 font-medium">Model</th>
                  <th className="text-left py-2 pr-4 font-medium">Result</th>
                  <th className="text-right py-2 pr-4 font-medium">L∞/255</th>
                  <th className="text-right py-2 pr-4 font-medium">SSIM</th>
                  <th className="text-right py-2 font-medium">PSNR</th>
                  <th className="py-2 w-16"></th>
                </tr>
              </thead>
              <tbody>
                {entries.map(e => (
                  <tr key={e.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 cursor-pointer" onClick={() => handleView(e.id)}>
                    <td className="py-2 pr-2" onClick={ev => ev.stopPropagation()}>
                      <input type="checkbox" checked={compareIds.includes(e.id)}
                        onChange={() => toggleCompare(e.id)}
                        className="w-3.5 h-3.5 rounded border-slate-600 accent-[var(--accent)]" />
                    </td>
                    <td className="py-2 pr-4 text-slate-500 whitespace-nowrap">{new Date(e.timestamp * 1000).toLocaleString()}</td>
                    <td className="py-2 pr-4">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${e.domain === 'audio' ? 'bg-purple-500/15 text-purple-400 border-purple-500/20' : 'bg-blue-500/15 text-blue-400 border-blue-500/20'}`}>
                        {e.domain || 'image'}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-slate-300 truncate max-w-[120px]" title={e.attack}>{e.attack}</td>
                    <td className="py-2 pr-4 text-slate-400 truncate max-w-[140px]" title={e.model}>{e.model?.split('/').pop()}</td>
                    <td className="py-2 pr-4">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${e.success ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20' : 'bg-red-500/15 text-red-400 border-red-500/20'}`}>
                        {e.success ? 'SUCCESS' : 'FAILED'}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-right text-slate-400">{e.metrics?.linf_255?.toFixed(1) ?? '—'}</td>
                    <td className="py-2 pr-4 text-right text-slate-400">{e.metrics?.ssim?.toFixed(3) ?? '—'}</td>
                    <td className="py-2 text-right text-slate-400">{e.metrics?.psnr?.toFixed(1) ?? '—'}</td>
                    <td className="py-2 text-right" onClick={ev => ev.stopPropagation()}>
                      <button onClick={() => handleDelete(e.id)} className="text-slate-600 hover:text-red-400 transition" title="Delete">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {pages > 1 && (
            <div className="flex items-center justify-center gap-2 mt-4 pt-4 border-t border-slate-800/50">
              <button disabled={page === 0} onClick={() => setPage(p => p - 1)}
                className="px-3 py-1 text-xs bg-slate-700/60 border border-slate-600 rounded-lg text-slate-400 hover:text-slate-200 disabled:opacity-30 transition">
                Prev
              </button>
              <span className="text-xs text-slate-500">Page {page + 1} of {pages}</span>
              <button disabled={page >= pages - 1} onClick={() => setPage(p => p + 1)}
                className="px-3 py-1 text-xs bg-slate-700/60 border border-slate-600 rounded-lg text-slate-400 hover:text-slate-200 disabled:opacity-30 transition">
                Next
              </button>
            </div>
          )}
        </Card>
      )}

      {/* Detail Modal */}
      {selected && <DetailModal entry={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}


function DetailModal({ entry, onClose }) {
  const e = entry
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div className="relative bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full max-h-[85vh] overflow-y-auto p-6" onClick={ev => ev.stopPropagation()}>
        <div className="flex justify-between items-start mb-6">
          <div>
            <h3 className="text-sm font-bold text-slate-200">{e.attack}</h3>
            <p className="text-xs text-slate-500">{e.model} — {new Date(e.timestamp * 1000).toLocaleString()}</p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-lg">&times;</button>
        </div>

        {/* Result badge */}
        <div className="mb-4">
          <span className={`px-3 py-1 rounded-full text-xs font-bold ${e.success ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
            {e.success ? 'ATTACK SUCCESSFUL' : 'ATTACK FAILED'}
          </span>
          <span className="ml-3 text-xs text-slate-500">
            {e.original_class} → {e.adversarial_class} (target: {e.target_class})
          </span>
        </div>

        {/* Metrics */}
        {e.metrics && (
          <div className="grid grid-cols-4 gap-3 mb-6">
            {Object.entries(e.metrics).map(([k, v]) => (
              <div key={k} className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3 text-center">
                <div className="text-sm font-bold text-slate-200">{typeof v === 'number' ? (v < 0.01 ? v.toExponential(2) : v.toFixed(4)) : v}</div>
                <div className="text-[10px] text-slate-600 uppercase tracking-wider mt-0.5">{k.replace(/_/g, ' ')}</div>
              </div>
            ))}
          </div>
        )}

        {/* Images */}
        <div className="grid grid-cols-2 gap-4">
          {e.adversarial_b64 && (
            <div>
              <span className="text-[10px] text-slate-500 block mb-1">Adversarial Output</span>
              <img src={`data:image/png;base64,${e.adversarial_b64}`} alt="Adversarial" className="w-full rounded-lg border border-slate-700" />
              <button onClick={() => downloadB64(e.adversarial_b64, `adversarial_${e.attack}.png`, 'image/png')}
                className="mt-2 text-xs text-[var(--accent)] hover:brightness-110">Download PNG</button>
            </div>
          )}
          {e.perturbation_b64 && (
            <div>
              <span className="text-[10px] text-slate-500 block mb-1">Perturbation (10x)</span>
              <img src={`data:image/png;base64,${e.perturbation_b64}`} alt="Perturbation" className="w-full rounded-lg border border-slate-700" />
              <button onClick={() => downloadB64(e.perturbation_b64, `perturbation_${e.attack}.png`, 'image/png')}
                className="mt-2 text-xs text-[var(--accent)] hover:brightness-110">Download PNG</button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/50 flex items-center justify-center mb-6">
        <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <h3 className="text-sm font-semibold text-slate-400 mb-2">No History Yet</h3>
      <p className="text-xs text-slate-600 max-w-xs text-center">
        Attack results are automatically saved here. Run an image or audio attack to get started.
      </p>
    </div>
  )
}
