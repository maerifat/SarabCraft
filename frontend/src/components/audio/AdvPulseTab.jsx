import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function AdvPulseTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [targetText, setTargetText] = useState('')
  const [dur, setDur] = useState(0.3)
  const [eps, setEps] = useState(0.1)
  const [iter, setIter] = useState(400)
  const [lr, setLr] = useState(0.005)
  const [physical, setPhysical] = useState('No')

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file || !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('target_text', targetText)
    fd.append('pulse_duration', dur); fd.append('epsilon', eps)
    fd.append('iterations', iter); fd.append('lr', lr)
    fd.append('physical', physical === 'Yes')
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <TextInput label="Target transcription" value={targetText} onChange={setTargetText} placeholder="Target text" />
        <Select label="Physical (over-the-air)" value={physical} onChange={setPhysical} options={['No', 'Yes']} />
      </InputGrid>
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Pulse duration (s)" value={dur} onChange={setDur} min={0.05} max={0.5} step={0.05} defaultValue={0.3} />
        <Slider label="Epsilon (L∞)" value={eps} onChange={setEps} min={0.01} max={0.3} step={0.01} defaultValue={0.1} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={1500} step={50} defaultValue={400} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.05} step={0.001} defaultValue={0.005} />
      </ParamGrid>
      <p className="text-[11px] text-slate-500 mt-2">Learns a subsecond, synchronization-free universal pulse injected at a random offset in the audio. Enable "Physical" for over-the-air robustness (room impulse + noise simulation).</p>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run AdvPulse Attack" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText })} />}
  </>)
}
