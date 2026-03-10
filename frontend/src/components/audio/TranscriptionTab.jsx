import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, AsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function TranscriptionTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [targetText, setTargetText] = useState('')
  const [eps, setEps] = useState(0.05)
  const [iter, setIter] = useState(300)
  const [lr, setLr] = useState(0.005)
  const [opt, setOpt] = useState('C&W (Adam)')

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file || !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('target_text', targetText)
    fd.append('epsilon', eps); fd.append('iterations', iter); fd.append('lr', lr); fd.append('optimizer', opt)
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <TextInput label="Target transcription" value={targetText} onChange={setTargetText} placeholder="e.g. Hello world" />
      </InputGrid>
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Epsilon" value={eps} onChange={setEps} min={0.005} max={0.3} step={0.005} defaultValue={0.05} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={2000} step={50} defaultValue={300} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.0005} max={0.05} step={0.0005} defaultValue={0.005} />
        <Select label="Optimizer" value={opt} onChange={setOpt} options={['C&W (Adam)', 'PGD (Sign-based)']} />
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Transcription Attack" /><ErrorMsg msg={error} /></div>
    {result && <AsrResult result={result} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText })} originalWavB64={origB64} />}
  </>)
}
