import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function SirenTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [mode, setMode] = useState('Targeted')
  const [targetText, setTargetText] = useState('')
  const [eps, setEps] = useState(0.05)
  const [particles, setParticles] = useState(25)
  const [iter, setIter] = useState(150)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file) return
    if (mode === 'Targeted' && !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('mode', mode)
    fd.append('target_text', targetText); fd.append('epsilon', eps)
    fd.append('n_particles', particles); fd.append('iterations', iter)
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <Select label="Mode" value={mode} onChange={setMode} options={['Targeted', 'Untargeted']} />
        {mode === 'Targeted' && <TextInput label="Target transcription" value={targetText} onChange={setTargetText} placeholder="Target text" />}
      </InputGrid>
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Epsilon (L∞)" value={eps} onChange={setEps} min={0.01} max={0.3} step={0.01} defaultValue={0.05} />
        <Slider label="Swarm particles" value={particles} onChange={setParticles} min={5} max={60} step={5} defaultValue={25} />
        <Slider label="PSO iterations" value={iter} onChange={setIter} min={20} max={500} step={10} defaultValue={150} />
      </ParamGrid>
      <p className="text-[11px] text-slate-500 mt-2">Gradient-free particle-swarm optimization. Purely score-based (black-box). Targeted forces the chosen transcript; untargeted degrades the correct output.</p>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run SirenAttack" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText })} />}
  </>)
}
