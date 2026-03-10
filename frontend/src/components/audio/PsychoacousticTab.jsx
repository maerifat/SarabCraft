import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function PsychoacousticTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [targetText, setTargetText] = useState('')
  const [iter, setIter] = useState(500)
  const [lr, setLr] = useState(0.005)
  const [mw, setMw] = useState(1.0)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file || !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('target_text', targetText)
    fd.append('iterations', iter); fd.append('lr', lr); fd.append('masking_weight', mw)
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <TextInput label="Target transcription" value={targetText} onChange={setTargetText} placeholder="Target text" />
      </InputGrid>
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={3000} step={50} defaultValue={500} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.05} step={0.001} defaultValue={0.005} />
        <Slider label="Masking weight" value={mw} onChange={setMw} min={0.1} max={10} step={0.1} defaultValue={1.0} />
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Psychoacoustic Attack" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText })} />}
  </>)
}
