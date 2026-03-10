import { useState, useMemo, useRef, useEffect } from 'react'
import { Card, SectionLabel } from '../ui/Section'

function LeaderboardCard({ summary, domain }) {
  if (!summary || !summary.best_attack) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '0.75rem' }} className="mb-4">
      <div className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 border border-amber-500/20 rounded-xl p-4 text-center">
        <div className="text-[10px] uppercase tracking-wider text-amber-500/70 mb-1">Best Attack</div>
        <div className="text-base font-black text-amber-400">{summary.best_attack}</div>
        {summary.best_params && (
          <div className="text-[10px] text-amber-500/60 mt-0.5">
            ε={summary.best_params.epsilon} · {summary.best_params.iterations} iters{domain === 'audio' && summary.best_params.lr != null ? ` · lr=${summary.best_params.lr}` : ''}
          </div>
        )}
      </div>
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 text-center">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Transfer Rate</div>
        <div className={`text-2xl font-black ${summary.best_transfer_rate > 0.5 ? 'text-emerald-400' : 'text-red-400'}`}>
          {(summary.best_transfer_rate * 100).toFixed(0)}%
        </div>
      </div>
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 text-center">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Combos</div>
        <div className="text-lg font-bold text-white">{summary.successful_combos || summary.total_combos}</div>
        {summary.failed_combos > 0 && <div className="text-[10px] text-red-400">{summary.failed_combos} failed</div>}
      </div>
    </div>
  )
}

function ParetoChart({ results, domain }) {
  const canvasRef = useRef(null)
  const [hoveredIdx, setHoveredIdx] = useState(null)
  const data = useMemo(() =>
    results.filter(r => !r.error).map(r => ({
      attack: r.attack,
      distortion: domain === 'image' ? (r.distortion?.l2 || 0) : (r.distortion?.snr_db || 0),
      rate: r.transfer_rate || 0,
      params: r.params,
    })), [results, domain]
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data.length) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width, h = canvas.height
    const pad = { t: 20, r: 20, b: 35, l: 45 }

    ctx.clearRect(0, 0, w, h)

    const dists = data.map(d => d.distortion)
    const minD = Math.min(...dists), maxD = Math.max(...dists)
    const rangeD = maxD - minD || 1
    const scaleX = d => pad.l + ((d - minD) / rangeD) * (w - pad.l - pad.r)
    const scaleY = r => pad.t + (1 - r) * (h - pad.t - pad.b)

    ctx.strokeStyle = 'rgba(100,116,139,0.15)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = scaleY(i / 4)
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke()
    }

    ctx.fillStyle = 'rgba(148,163,184,0.6)'
    ctx.font = '10px system-ui'
    ctx.textAlign = 'center'
    ctx.fillText(domain === 'image' ? 'Distortion (L2)' : 'SNR (dB)', w / 2, h - 4)
    ctx.save()
    ctx.translate(10, h / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Transfer Rate', 0, 0)
    ctx.restore()

    ctx.textAlign = 'center'
    for (let i = 0; i <= 4; i++) {
      const v = minD + (rangeD * i) / 4
      ctx.fillText(v.toFixed(1), scaleX(v), h - pad.b + 14)
    }
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      ctx.fillText(`${(i * 25)}%`, pad.l - 6, scaleY(i / 4) + 3)
    }

    const attacks = [...new Set(data.map(d => d.attack))]
    const colors = ['#22d3ee', '#a78bfa', '#f472b6', '#34d399', '#fbbf24', '#fb7185', '#60a5fa', '#c084fc']

    data.forEach((d, i) => {
      const x = scaleX(d.distortion), y = scaleY(d.rate)
      const ci = attacks.indexOf(d.attack) % colors.length
      ctx.beginPath()
      ctx.arc(x, y, hoveredIdx === i ? 7 : 5, 0, Math.PI * 2)
      ctx.fillStyle = hoveredIdx === i ? colors[ci] : colors[ci] + '99'
      ctx.fill()
      ctx.strokeStyle = colors[ci]
      ctx.lineWidth = 1.5
      ctx.stroke()
    })
  }, [data, domain, hoveredIdx])

  const handleMouseMove = e => {
    const canvas = canvasRef.current
    if (!canvas || !data.length) return
    const rect = canvas.getBoundingClientRect()
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width)
    const my = (e.clientY - rect.top) * (canvas.height / rect.height)

    const pad = { t: 20, r: 20, b: 35, l: 45 }
    const w = canvas.width, h = canvas.height
    const dists = data.map(d => d.distortion)
    const minD = Math.min(...dists), maxD = Math.max(...dists)
    const rangeD = maxD - minD || 1
    const scaleX = d => pad.l + ((d - minD) / rangeD) * (w - pad.l - pad.r)
    const scaleY = r => pad.t + (1 - r) * (h - pad.t - pad.b)

    let closest = null, closestDist = 20
    data.forEach((d, i) => {
      const dx = scaleX(d.distortion) - mx, dy = scaleY(d.rate) - my
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < closestDist) { closest = i; closestDist = dist }
    })
    setHoveredIdx(closest)
  }

  return (
    <div className="relative">
      <canvas ref={canvasRef} width={400} height={220} className="w-full h-auto rounded-lg bg-slate-900/50 border border-slate-700/30"
        onMouseMove={handleMouseMove} onMouseLeave={() => setHoveredIdx(null)} />
      {hoveredIdx !== null && data[hoveredIdx] && (
        <div className="absolute top-2 right-2 bg-slate-800/95 border border-slate-700/50 rounded-lg px-2.5 py-1.5 text-[10px] backdrop-blur-sm pointer-events-none">
          <div className="text-[var(--accent)] font-semibold">{data[hoveredIdx].attack}</div>
          <div className="text-slate-400">
            {domain === 'image' ? `L2: ${data[hoveredIdx].distortion.toFixed(3)}` : `SNR: ${data[hoveredIdx].distortion.toFixed(1)} dB`}
            {' · '}{(data[hoveredIdx].rate * 100).toFixed(0)}% transfer
          </div>
        </div>
      )}
    </div>
  )
}

function TransferHeatmap({ results }) {
  const data = useMemo(() => {
    const valid = results.filter(r => !r.error && r.transfer_results?.length)
    if (!valid.length) return null
    const attacks = valid.map(r => `${r.attack} (ε=${r.params?.epsilon || '?'})`)
    const targets = [...new Set(valid.flatMap(r => r.transfer_results.map(t => t.target)))]
    const matrix = valid.map(r => targets.map(t => {
      const tr = r.transfer_results.find(x => x.target === t)
      if (!tr || tr.error) return 'none'
      return tr.success ? 'success' : 'fail'
    }))
    return { attacks, targets, matrix }
  }, [results])

  if (!data) return null

  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] border-collapse">
        <thead>
          <tr>
            <th className="text-left py-1 pr-2 text-slate-500 font-normal sticky left-0 bg-slate-900/80"></th>
            {data.targets.map(t => (
              <th key={t} className="py-1 px-1 text-slate-500 font-normal whitespace-nowrap max-w-[80px] truncate" title={t}>
                {t.replace(/^\[.*?\]\s*/, '').split('/').pop()?.slice(0, 12)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.attacks.map((atk, ri) => (
            <tr key={ri}>
              <td className="py-0.5 pr-2 text-slate-400 whitespace-nowrap sticky left-0 bg-slate-900/80">{atk}</td>
              {data.matrix[ri].map((val, ci) => (
                <td key={ci} className="py-0.5 px-1">
                  <div className={`w-5 h-5 rounded-sm ${
                    val === 'success' ? 'bg-emerald-500/60' : val === 'fail' ? 'bg-red-500/40' : 'bg-slate-700/30'
                  }`} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ResultsTable({ results, domain, onExpand, expandedIdx }) {
  const [sortKey, setSortKey] = useState('transfer_rate')
  const [sortDir, setSortDir] = useState(-1)

  const toggleSort = key => {
    if (sortKey === key) setSortDir(d => -d)
    else { setSortKey(key); setSortDir(-1) }
  }

  const sorted = useMemo(() => {
    const arr = [...results]
    arr.sort((a, b) => {
      let va = a, vb = b
      if (sortKey === 'transfer_rate') { va = a.transfer_rate || 0; vb = b.transfer_rate || 0 }
      else if (sortKey === 'distortion') {
        va = domain === 'image' ? (a.distortion?.l2 || 999) : (a.distortion?.snr_db || 0)
        vb = domain === 'image' ? (b.distortion?.l2 || 999) : (b.distortion?.snr_db || 0)
      }
      else if (sortKey === 'elapsed') { va = a.elapsed_ms || 0; vb = b.elapsed_ms || 0 }
      else if (sortKey === 'attack') return sortDir * a.attack.localeCompare(b.attack)
      return sortDir * (va - vb)
    })
    return arr
  }, [results, sortKey, sortDir, domain])

  const SortHeader = ({ k, children, className = '' }) => (
    <th className={`py-2 px-3 cursor-pointer select-none hover:text-[var(--accent)] transition ${className}`}
      onClick={() => toggleSort(k)}>
      {children} {sortKey === k && <span className="text-[var(--accent)]">{sortDir > 0 ? '↑' : '↓'}</span>}
    </th>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-slate-700/50 text-[10px] uppercase tracking-wider">
            <th className="py-2 px-3 text-left w-8">#</th>
            <SortHeader k="attack" className="text-left">Attack</SortHeader>
            <th className="py-2 px-3 text-left">Params</th>
            <th className="py-2 px-3 text-left">{domain === 'image' ? 'Source Prediction' : 'Source Result'}</th>
            <SortHeader k="distortion" className="text-right">{domain === 'image' ? 'L2' : 'SNR'}</SortHeader>
            <SortHeader k="transfer_rate" className="text-right">Transfer</SortHeader>
            <th className="py-2 px-3 text-center">Detail</th>
            <SortHeader k="elapsed" className="text-right">Time</SortHeader>
            <th className="py-2 px-3 text-center w-10"></th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <ResultRow key={`${r.attack}-${i}`} r={r} rank={i + 1} domain={domain}
              expanded={expandedIdx === r.index} onExpand={() => onExpand(r.index)} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ChevronIcon({ expanded }) {
  return (
    <svg className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`}
      fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  )
}

function ResultRow({ r, rank, domain, expanded, onExpand }) {
  if (r.error) {
    return (
      <tr className="border-b border-slate-800/40 text-slate-500">
        <td className="py-2.5 px-3">{rank}</td>
        <td className="py-2.5 px-3">{r.attack}</td>
        <td colSpan={7} className="py-2.5 px-3 text-red-400/70 text-[10px]">Error: {r.error.slice(0, 80)}</td>
      </tr>
    )
  }

  const trDetail = r.transfer_results || []
  const pStr = domain === 'image'
    ? `ε=${r.params?.epsilon || '?'} · ${r.params?.iterations || '?'} iters`
    : `ε=${r.params?.epsilon || '?'} · ${r.params?.iterations || '?'} iters${r.params?.lr != null ? ` · lr=${r.params.lr}` : ''}`
  const sourceValue = domain === 'image' ? (r.source_class || '?') : (r.result_text || '(silence)')
  const shortSourceValue = sourceValue.length > 28 ? sourceValue.slice(0, 26) + '…' : sourceValue
  const sourceHint = domain === 'audio'
    ? (r.evaluation_mode === 'untargeted'
        ? (r.original_text ? `Original: "${r.original_text}"` : 'Untargeted disruption')
        : (r.target_text ? `Target: "${r.target_text}"` : 'Targeted attack'))
    : ''

  return (
    <>
      <tr className={`border-b border-slate-800/40 transition group cursor-pointer
        ${expanded ? 'bg-slate-800/30' : 'hover:bg-slate-800/20'}`}
        onClick={onExpand}>
        <td className="py-2.5 px-3 text-slate-500">{rank}</td>
        <td className="py-2.5 px-3 text-slate-200 font-medium">{r.attack}</td>
        <td className="py-2.5 px-3 text-[10px] text-slate-400">{pStr}</td>
        <td className="py-2.5 px-3 text-left">
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${r.source_success ? 'bg-emerald-400' : 'bg-red-400'}`} />
            <div className="min-w-0">
              <span className={`text-[10px] truncate block max-w-[160px] ${r.source_success ? 'text-emerald-400' : 'text-slate-400'}`}
                title={sourceValue}>
                {shortSourceValue}
              </span>
              {sourceHint && <span className="text-[10px] text-slate-600 truncate block max-w-[160px]">{sourceHint}</span>}
            </div>
          </div>
        </td>
        <td className="py-2.5 px-3 text-right text-slate-400">
          {domain === 'image' ? r.distortion?.l2?.toFixed(3) : r.distortion?.snr_db?.toFixed(1)}
        </td>
        <td className="py-2.5 px-3 text-right">
          <span className={`font-bold ${(r.transfer_rate || 0) > 0.5 ? 'text-emerald-400' : 'text-red-400'}`}>
            {((r.transfer_rate || 0) * 100).toFixed(0)}%
          </span>
          <span className="text-slate-600 ml-1 text-[10px]">({r.transfer_success}/{r.transfer_tested})</span>
        </td>
        <td className="py-2.5 px-3">
          <div className="flex gap-0.5 justify-center">
            {trDetail.map((t, ti) => (
              <div key={ti} className={`w-3 h-3 rounded-sm ${t.error ? 'bg-slate-600' : t.success ? 'bg-emerald-500/70' : 'bg-red-500/50'}`}
                title={domain === 'image'
                  ? `${t.target}: ${t.error ? 'error' : t.success ? 'fooled' : 'resisted'}${t.predictions?.length ? ` — top: ${t.predictions[0].label}` : ''}`
                  : `${t.target}: ${t.error ? 'error' : t.success ? (r.evaluation_mode === 'untargeted' ? 'changed' : 'fooled') : (r.evaluation_mode === 'untargeted' ? 'preserved' : 'resisted')}${t.transcription ? ` — "${t.transcription}"` : ''}`} />
            ))}
          </div>
        </td>
        <td className="py-2.5 px-3 text-right text-slate-500 text-[10px]">{(r.elapsed_ms / 1000).toFixed(1)}s</td>
        <td className="py-2.5 px-3 text-center">
          <ChevronIcon expanded={expanded} />
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={9} className="bg-slate-800/20 border-b border-slate-700/30">
            <div className="px-5 py-4">
              <ExpandedRow r={r} domain={domain} />
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

function ExpandedRow({ r, domain }) {
  return (
    <div className="space-y-3">
      {/* Source model result */}
      {domain === 'image' && (
        <div className="flex items-start gap-6 text-[10px]">
          <div>
            <span className="text-slate-500 block mb-0.5">Source Model Result</span>
            <span className={r.source_success ? 'text-emerald-400 font-medium' : 'text-amber-400 font-medium'}>
              {r.source_success ? 'Targeted attack succeeded' : 'Targeted attack missed'}
            </span>
            <span className="text-slate-500 ml-1">
              — classified as <span className="text-slate-300">"{r.source_class || '?'}"</span>
              {r.original_class && <> (was "{r.original_class}")</>}
            </span>
          </div>
        </div>
      )}
      {domain === 'audio' && (
        <div className="flex items-start gap-6 text-[10px]">
          <div>
            <span className="text-slate-500 block mb-0.5">Source Model Result</span>
            <span className={r.source_success ? 'text-emerald-400 font-medium' : 'text-amber-400 font-medium'}>
              {r.evaluation_mode === 'untargeted'
                ? (r.source_success ? 'Untargeted disruption succeeded' : 'Transcription stayed close to the original')
                : (r.source_success ? 'Targeted attack succeeded' : 'Targeted attack missed')}
            </span>
            <span className="text-slate-500 ml-1">
              — output <span className="text-slate-300">"{r.result_text || '(silence)'}"</span>
              {r.original_text && <> from original <span className="text-slate-300">"{r.original_text}"</span></>}
              {r.target_text && <> toward target <span className="text-slate-300">"{r.target_text}"</span></>}
            </span>
          </div>
        </div>
      )}

      {r.adversarial_b64 && domain === 'image' && (
        <div>
          <span className="text-[10px] text-slate-500 mb-1 block">Adversarial preview</span>
          <img src={`data:image/png;base64,${r.adversarial_b64}`} alt="adversarial" className="w-32 h-32 object-cover rounded-lg border border-slate-700/50" />
        </div>
      )}
      {r.adversarial_b64 && domain === 'audio' && (
        <div>
          <span className="text-[10px] text-slate-500 mb-1 block">Adversarial audio</span>
          <audio controls src={`data:audio/wav;base64,${r.adversarial_b64}`} className="h-8 w-full max-w-sm" />
        </div>
      )}

      {r.distortion && (
        <div className="flex gap-4">
          {Object.entries(r.distortion).map(([k, v]) => (
            <div key={k} className="text-[10px]">
              <span className="text-slate-500 uppercase">{k}:</span>{' '}
              <span className="text-slate-300">{typeof v === 'number' ? v.toFixed(3) : v}</span>
            </div>
          ))}
        </div>
      )}

      {r.transfer_results?.length > 0 && (
        <div>
          <span className="text-[10px] text-slate-500 mb-1.5 block">Transfer results</span>
          <div className="space-y-1.5">
            {r.transfer_results.map((t, i) => (
              <div key={i} className="flex items-start gap-2 text-[10px]">
                <span className={`w-2 h-2 rounded-full shrink-0 mt-0.5 ${t.error ? 'bg-slate-600' : t.success ? 'bg-emerald-500' : 'bg-red-500'}`} />
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-300 font-medium">{t.target}</span>
                    {t.error ? (
                      <span className="text-red-400/70">{t.error.slice(0, 40)}</span>
                    ) : (
                      <span className={t.success ? 'text-emerald-400' : 'text-red-400'}>
                        {domain === 'audio'
                          ? (t.success
                              ? (r.evaluation_mode === 'untargeted' ? 'Changed' : 'Fooled')
                              : (r.evaluation_mode === 'untargeted' ? 'Preserved' : 'Resisted'))
                          : (t.success ? 'Fooled' : 'Resisted')}
                        {t.transcription && ` — "${t.transcription.slice(0, 30)}"`}
                        {t.original_transcription && domain === 'audio' && ` vs "${t.original_transcription.slice(0, 30)}"`}
                        {t.confidence_drop > 0 && ` (conf drop: ${(t.confidence_drop * 100).toFixed(1)}%)`}
                      </span>
                    )}
                  </div>
                  {t.predictions?.length > 0 && !t.error && (
                    <div className="text-slate-500 mt-0.5">
                      Top: {t.predictions.slice(0, 3).map((p, pi) => (
                        <span key={pi}>
                          {pi > 0 && ', '}
                          <span className="text-slate-400">{p.label}</span>
                          <span className="text-slate-600 ml-0.5">({(p.confidence * 100).toFixed(1)}%)</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {r.adversarial_b64 && (
        <a href={`data:${domain === 'image' ? 'image/png' : 'audio/wav'};base64,${r.adversarial_b64}`}
          download={`adversarial_${r.attack}_${r.index}.${domain === 'image' ? 'png' : 'wav'}`}
          className="inline-block text-[10px] text-[var(--accent)] hover:brightness-110 underline">
          Download adversarial
        </a>
      )}
    </div>
  )
}

function exportCSV(results, domain) {
  const sorted = [...results].sort((a, b) => (b.transfer_rate || 0) - (a.transfer_rate || 0))

  const allTargets = [...new Set(
    sorted.flatMap(r => (r.transfer_results || []).map(t => t.target))
  )]

  const esc = v => {
    const s = String(v ?? '')
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? `"${s.replace(/"/g, '""')}"` : s
  }

  const headers = domain === 'image'
    ? [
        'Rank', 'Attack', 'Epsilon', 'Iterations', 'Alpha',
        'Source_Success', 'Source_Predicted_Class', 'Original_Class',
        'L2', 'SSIM', 'PSNR',
        'Transfer_Rate_Pct', 'Transfer_Success', 'Transfer_Tested',
        ...allTargets.flatMap(t => {
          const short = t.split('/').pop() || t
          return [`${short}_Success`, `${short}_TopPred`, `${short}_TopConf`, `${short}_ConfDrop`]
        }),
        'Elapsed_ms', 'Error',
      ]
    : [
        'Rank', 'Attack', 'Epsilon', 'Iterations', 'Learning_Rate',
        'Source_Success', 'Evaluation_Mode', 'Source_Result_Text', 'Original_Text', 'Target_Text',
        'SNR_dB',
        'Transfer_Rate_Pct', 'Transfer_Success', 'Transfer_Tested',
        ...allTargets.flatMap(t => {
          const short = t.split('/').pop() || t
          return [`${short}_Success`, `${short}_Transcription`, `${short}_Baseline`, `${short}_WER_Pct`]
        }),
        'Elapsed_ms', 'Error',
      ]

  const rows = sorted.map((r, i) => {
    if (domain === 'image') {
      const base = [
        i + 1,
        esc(r.attack),
        r.params?.epsilon ?? '',
        r.params?.iterations ?? '',
        r.params?.alpha ?? '',
        r.error ? '' : (r.source_success ? 'Yes' : 'No'),
        esc(r.source_class || ''),
        esc(r.original_class || ''),
      ]

      const distortion = [
        r.distortion?.l2?.toFixed(4) ?? '',
        r.distortion?.ssim?.toFixed(4) ?? '',
        r.distortion?.psnr?.toFixed(1) ?? '',
      ]

      const transfer = [
        r.error ? '' : ((r.transfer_rate || 0) * 100).toFixed(1),
        r.transfer_success ?? '',
        r.transfer_tested ?? '',
      ]

      const perTarget = allTargets.flatMap(tName => {
        const tr = (r.transfer_results || []).find(t => t.target === tName)
        if (!tr) return ['', '', '', '']
        if (tr.error) return ['Error', esc(tr.error), '', '']
        return [
          tr.success ? 'Yes' : 'No',
          esc(tr.predictions?.[0]?.label || ''),
          tr.predictions?.[0]?.confidence != null ? (tr.predictions[0].confidence * 100).toFixed(1) : '',
          tr.confidence_drop != null ? (tr.confidence_drop * 100).toFixed(1) : '',
        ]
      })

      return [...base, ...distortion, ...transfer, ...perTarget, r.elapsed_ms ?? '', esc(r.error || '')]
    }

    const base = [
      i + 1,
      esc(r.attack),
      r.params?.epsilon ?? '',
      r.params?.iterations ?? '',
      r.params?.lr ?? '',
      r.error ? '' : (r.source_success ? 'Yes' : 'No'),
      esc(r.evaluation_mode || 'targeted'),
      esc(r.result_text || ''),
      esc(r.original_text || ''),
      esc(r.target_text || ''),
    ]

    const distortion = [r.distortion?.snr_db?.toFixed(1) ?? '']
    const transfer = [
      r.error ? '' : ((r.transfer_rate || 0) * 100).toFixed(1),
      r.transfer_success ?? '',
      r.transfer_tested ?? '',
    ]

    const perTarget = allTargets.flatMap(tName => {
      const tr = (r.transfer_results || []).find(t => t.target === tName)
      if (!tr) return ['', '', '', '']
      if (tr.error) return ['Error', esc(tr.error), '', '']
      return [
        tr.success ? 'Yes' : 'No',
        esc(tr.transcription || ''),
        esc(tr.original_transcription || ''),
        tr.wer != null ? (tr.wer * 100).toFixed(1) : '',
      ]
    })

    return [...base, ...distortion, ...transfer, ...perTarget, r.elapsed_ms ?? '', esc(r.error || '')]
  })

  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `benchmark_${domain}_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')}.csv`
  a.click()
}

export default function BenchmarkResults({ results, summary, total, elapsed, domain, running, onAbort }) {
  const [expandedIdx, setExpandedIdx] = useState(null)
  const autoExpandedRef = useRef(false)
  const completed = results.length

  useEffect(() => {
    if (results.length > 0 && !autoExpandedRef.current) {
      autoExpandedRef.current = true
      setExpandedIdx(results[0].index)
    }
  }, [results])

  useEffect(() => {
    if (results.length === 0) {
      autoExpandedRef.current = false
      setExpandedIdx(null)
    }
  }, [results.length === 0])

  return (
    <div className="space-y-4">
      {summary && (
        <div className="flex items-center justify-end mb-1">
          <button onClick={() => exportCSV(results, domain)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/40 text-slate-300 border border-slate-600/30 hover:bg-slate-700/60 transition">
            Export CSV
          </button>
        </div>
      )}

      {summary && <LeaderboardCard summary={summary} domain={domain} />}

      {completed >= 2 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem' }}>
          <Card>
            <SectionLabel>Distortion vs Transfer Rate</SectionLabel>
            <ParetoChart results={results} domain={domain} />
          </Card>
          <Card>
            <SectionLabel>Transfer Heatmap</SectionLabel>
            <TransferHeatmap results={results} />
          </Card>
        </div>
      )}

      {completed > 0 && (
        <Card>
          <SectionLabel>All Results ({completed})</SectionLabel>
          <ResultsTable results={results} domain={domain}
            expandedIdx={expandedIdx}
            onExpand={idx => setExpandedIdx(expandedIdx === idx ? null : idx)} />
        </Card>
      )}
    </div>
  )
}
