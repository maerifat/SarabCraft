import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function OTATab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [targetText, setTargetText] = useState('')
  const [eps, setEps] = useState(0.08)
  const [iter, setIter] = useState(500)
  const [lr, setLr] = useState(0.005)
  const [rooms, setRooms] = useState(3)
  const [snr, setSnr] = useState(20)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file || !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('target_text', targetText)
    fd.append('epsilon', eps); fd.append('iterations', iter); fd.append('lr', lr)
    fd.append('n_rooms', rooms); fd.append('noise_snr_db', snr)
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
        <Slider label="Epsilon" value={eps} onChange={setEps} min={0.01} max={0.3} step={0.01} defaultValue={0.08} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={3000} step={50} defaultValue={500} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.05} step={0.001} defaultValue={0.005} />
        <Slider label="Room simulations" value={rooms} onChange={setRooms} min={1} max={5} step={1} defaultValue={3} />
        <Slider label="Noise SNR (dB)" value={snr} onChange={setSnr} min={10} max={50} step={5} defaultValue={20} />
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Over-the-Air Attack" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText })} />}
  </>)
}
