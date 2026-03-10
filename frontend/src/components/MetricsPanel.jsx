export default function MetricsPanel({ metrics }) {
  if (!metrics) return null

  const items = [
    { key: 'l2', label: 'L2 (RMS)', desc: 'Root mean square perturbation', good: v => v < 0.01, bad: v => v > 0.05 },
    { key: 'linf_255', label: 'L∞ /255', desc: 'Max pixel change', good: v => v < 8, bad: v => v > 32 },
    { key: 'ssim', label: 'SSIM', desc: 'Structural similarity', good: v => v > 0.95, bad: v => v < 0.8 },
    { key: 'psnr', label: 'PSNR (dB)', desc: 'Peak signal-to-noise', good: v => v > 40, bad: v => v < 25 },
    { key: 'l0_pixels', label: 'L0', desc: 'Pixels changed', good: v => v < 100, bad: v => v > 10000 },
    { key: 'mse', label: 'MSE', desc: 'Mean squared error', good: v => v < 0.001, bad: v => v > 0.01 },
  ]

  return (
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-semibold">Perturbation Quality</p>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        {items.map(({ key, label, desc, good, bad }) => {
          const v = metrics[key]
          if (v === undefined || v === null) return null
          const quality = good(v) ? 'good' : bad(v) ? 'bad' : 'mid'
          const color = quality === 'good' ? 'text-emerald-400 border-emerald-500/20' :
                        quality === 'bad' ? 'text-red-400 border-red-500/20' :
                        'text-amber-400 border-amber-500/20'
          return (
            <div key={key} className={`bg-slate-800/60 border rounded-lg p-2.5 text-center ${color}`} title={desc}>
              <div className="text-sm font-bold">{typeof v === 'number' ? (v < 0.001 && v > 0 ? v.toExponential(1) : v >= 100 ? v.toFixed(0) : v.toFixed(3)) : v}</div>
              <div className="text-[8px] text-slate-500 uppercase tracking-wider mt-0.5">{label}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
