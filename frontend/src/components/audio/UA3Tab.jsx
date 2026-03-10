import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, AudioPlayer, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function UA3Tab({ models: ua3Models, asrKeys = [], loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [targetText, setTargetText] = useState('')
  const [sel, setSel] = useState('Whisper + Wav2Vec2')
  const [iter, setIter] = useState(1500)
  const [eps, setEps] = useState(0.08)
  const [lr, setLr] = useState(0.005)
  const opts = ua3Models.length ? ua3Models : ['Whisper (base.en)', 'Wav2Vec2', 'HuBERT', 'Whisper + Wav2Vec2', 'Whisper + Wav2Vec2 + HuBERT']

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file || !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('target_text', targetText); fd.append('model_selection', sel)
    fd.append('iterations', iter); fd.append('epsilon', eps); fd.append('lr', lr)
    await onRun(fd)
  }

  const previewModel = asrKeys.length > 0 ? (asrKeys[0]?.value || asrKeys[0]) : null

  return (<>
    <Card><SectionLabel>Input — UA3: Universal Audio Adversarial Attack</SectionLabel>
      <p className="text-xs text-slate-500 mb-3">One perturbation that fools ALL selected ASR architectures simultaneously.</p>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="Model ensemble" value={sel} onChange={setSel} options={opts} />
        <TextInput label="Target transcription (all models)" value={targetText} onChange={setTargetText} placeholder="e.g. open the door" />
      </InputGrid>
      {previewModel && <div className="mt-3"><PreviewTranscription file={file} model={previewModel} /></div>}
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Iterations" value={iter} onChange={setIter} min={100} max={3000} step={100} defaultValue={1500} />
        <Slider label="L∞ budget (ε)" value={eps} onChange={setEps} min={0.01} max={0.2} step={0.01} defaultValue={0.08} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.02} step={0.001} defaultValue={0.005} />
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run UA3 Attack" /><ErrorMsg msg={error} /></div>
    {result && (
      <Card className="border-[var(--accent)]/30">
        <SectionLabel>Results</SectionLabel>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <p className={`text-lg font-bold mb-3 ${result.all_match ? 'text-green-400' : 'text-amber-400'}`}>{result.all_match ? 'ALL MODELS FOOLED' : 'Partial success'}</p>
            <div className="space-y-2">
              {result.per_model_results?.map((m, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${m.matched ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span className="text-slate-300 font-medium">{m.name}:</span>
                  <span className="text-slate-400 truncate">"{m.text}"</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-3">SNR: {result.snr_db?.toFixed(1)} dB | L∞: {result.linf?.toFixed(4)} | Models: {result.num_models}</p>
          </div>
          <div className="space-y-3">
            {origB64 && <AudioPlayer wavB64={origB64} label="Original audio" filename="original_audio.wav" />}
            <AudioPlayer wavB64={result.adversarial_wav_b64} label="Adversarial audio" />
            {onTransfer && <button onClick={() => onTransfer({ originalWavB64: origB64, targetText })} className="px-4 py-2 bg-[var(--accent-dim)] border border-[var(--accent)] text-[var(--accent)] font-semibold rounded-lg text-sm hover:bg-cyan-500/20 transition">Test Transfer</button>}
          </div>
        </div>
      </Card>
    )}
  </>)
}
