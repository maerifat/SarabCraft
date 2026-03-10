import { useState, useEffect } from 'react'
import { getModels, classifyImage } from '../api/client'
import { Card, SectionLabel, Select, RunButton, ErrorMsg } from './ui/Section'
import ModelPicker from './ui/ModelPicker'

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

export default function ClassifierTab() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('')
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getModels().then(d => {
      const items = normalizeModelItems(d)
      setModels(items)
      if (items.length > 0) setModel(prev => prev || d.default_id || items[0].value)
    }).catch(console.error)
  }, [])

  const pick = e => {
    const f = e.target.files?.[0]; setFile(f)
    if (f) { const r = new FileReader(); r.onload = ev => setPreview(ev.target.result); r.readAsDataURL(f) }
    else setPreview(null)
  }

  const handleClassify = async () => {
    if (!file) { setError('Upload an image'); return }
    setLoading(true); setError(''); setResult(null)
    try {
      const fd = new FormData(); fd.append('image_file', file); fd.append('model', model)
      setResult(await classifyImage(fd))
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-bold text-slate-200">Image Classification</h2>
        <p className="text-xs text-slate-500">Upload an image and classify it using a selected model</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', alignItems: 'start' }}>
        <Card>
          <SectionLabel>Input</SectionLabel>
          <div className="mb-3">
            <label className="block aspect-square max-w-[180px] border-2 border-dashed border-slate-600 hover:border-[var(--accent)] rounded-lg cursor-pointer overflow-hidden bg-slate-800/60 transition-colors relative">
              {preview ? <img src={preview} alt="" className="w-full h-full object-cover" /> : <div className="flex items-center justify-center h-full text-slate-500 text-xs">Click to upload</div>}
              <input type="file" accept="image/*" onChange={pick} className="absolute inset-0 opacity-0 cursor-pointer" />
            </label>
          </div>
          <ModelPicker label="Model" value={model} onChange={setModel} models={models} />
        </Card>

        {result && (
          <Card className="border-[var(--accent)]/30">
            <SectionLabel>Result</SectionLabel>
            <p className="text-2xl font-bold text-[var(--accent)]">{result.class}</p>
            <p className="text-slate-400 text-sm mt-1 mb-4">Confidence: {result.confidence?.toFixed(1)}%</p>
            {result.predictions && (
              <div className="space-y-1.5">
                {Object.entries(result.predictions).slice(0, 5).map(([k, v]) => (
                  <div key={k} className="flex items-center gap-2">
                    <div className="w-3/5 bg-slate-700/80 rounded-full h-3.5 overflow-hidden">
                      <div className={`h-full rounded-full ${k === result.class ? 'bg-[var(--accent)]' : 'bg-blue-500/50'}`} style={{ width: `${Math.min(v * 100, 100)}%` }} />
                    </div>
                    <span className={`text-[11px] truncate ${k === result.class ? 'text-slate-200 font-medium' : 'text-slate-400'}`}>{k}: {(v * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        )}
      </div>

      <div className="flex items-center gap-4">
        <RunButton onClick={handleClassify} loading={loading} label="Classify" loadingLabel="Classifying…" />
        <ErrorMsg msg={error} />
      </div>
    </div>
  )
}
