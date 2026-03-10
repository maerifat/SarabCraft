import { useState } from 'react'
import { compareGradCAM } from '../api/client'

export default function GradCAMPanel({ originalB64, adversarialB64, model }) {
  const [overlays, setOverlays] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(false)

  const run = async () => {
    if (overlays) { setExpanded(!expanded); return }
    setLoading(true); setError('')
    try {
      const res = await compareGradCAM({
        model: model || 'microsoft/resnet-50',
        original_b64: originalB64,
        adversarial_b64: adversarialB64,
      })
      setOverlays(res)
      setExpanded(true)
    } catch (e) {
      setError(e.message || 'GradCAM failed')
    } finally { setLoading(false) }
  }

  return (
    <div>
      <button onClick={run} disabled={loading}
        className="flex items-center gap-2 px-4 py-2 bg-purple-500/10 border border-purple-500/20 text-purple-400 rounded-lg text-xs font-medium hover:bg-purple-500/20 transition disabled:opacity-40">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        {loading ? 'Generating GradCAM...' : overlays ? (expanded ? 'Hide Attention Maps' : 'Show Attention Maps') : 'Generate GradCAM Overlays'}
      </button>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}

      {expanded && overlays && (
        <div className="mt-3 grid grid-cols-2 gap-3">
          {overlays.original_overlay && (
            <div>
              <span className="text-[10px] text-slate-500 mb-1 block">Original — Attention Map</span>
              <img src={`data:image/png;base64,${overlays.original_overlay}`} alt="Original GradCAM"
                className="w-full rounded-lg border border-slate-700" />
            </div>
          )}
          {overlays.adversarial_overlay && (
            <div>
              <span className="text-[10px] text-slate-500 mb-1 block">Adversarial — Attention Map</span>
              <img src={`data:image/png;base64,${overlays.adversarial_overlay}`} alt="Adversarial GradCAM"
                className="w-full rounded-lg border border-slate-700" />
            </div>
          )}
          <p className="col-span-2 text-[10px] text-slate-600">
            GradCAM highlights which regions the model focuses on. Red = high attention, blue = low attention.
            Compare original vs adversarial to see how the attack redirects the model's focus.
          </p>
        </div>
      )}
    </div>
  )
}
