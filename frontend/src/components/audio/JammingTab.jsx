import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function JammingTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [method, setMethod] = useState('Untargeted (Max CE)')
  const [eps, setEps] = useState(0.05)
  const [iter, setIter] = useState(300)
  const [lr, setLr] = useState(0.005)
  const [bLow, setBLow] = useState(300)
  const [bHigh, setBHigh] = useState(3400)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('method', method)
    fd.append('epsilon', eps); fd.append('iterations', iter); fd.append('lr', lr)
    fd.append('band_low_hz', bLow); fd.append('band_high_hz', bHigh)
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <Select label="Jamming method" value={method} onChange={setMethod} options={['Untargeted (Max CE)', 'Band-Limited Noise']} />
      </InputGrid>
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Epsilon" value={eps} onChange={setEps} min={0.01} max={0.3} step={0.01} defaultValue={0.05} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={2000} step={50} defaultValue={300} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.05} step={0.001} defaultValue={0.005} />
        {method === 'Band-Limited Noise' && <>
          <Slider label="Band low (Hz)" value={bLow} onChange={setBLow} min={100} max={1000} step={50} defaultValue={300} />
          <Slider label="Band high (Hz)" value={bHigh} onChange={setBHigh} min={1000} max={8000} step={100} defaultValue={3400} />
        </>}
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Speech Jamming" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} label="Jammed audio" originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText: '' })} />}
  </>)
}
