import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function TaskControlTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [task, setTask] = useState('translate')
  const [language, setLanguage] = useState('')
  const [dur, setDur] = useState(0.64)
  const [iter, setIter] = useState(250)
  const [lr, setLr] = useState(0.01)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('task', task)
    fd.append('language', language); fd.append('segment_duration', dur)
    fd.append('iterations', iter); fd.append('lr', lr)
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <Select label="Force task" value={task} onChange={setTask} options={['translate', 'transcribe']} />
        <TextInput label="Force language (optional)" value={language} onChange={setLanguage} placeholder="e.g. de, fr, es" />
      </InputGrid>
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Prefix duration (s)" value={dur} onChange={setDur} min={0.2} max={2.0} step={0.02} defaultValue={0.64} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={1000} step={50} defaultValue={250} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.05} step={0.001} defaultValue={0.01} />
      </ParamGrid>
      <p className="text-[11px] text-slate-500 mt-2">Learns a universal audio prefix that hijacks Whisper's task selection (e.g. forces translation instead of transcription). Requires a multilingual Whisper checkpoint (not a *.en model).</p>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Task-Control Attack" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText: '' })} />}
  </>)
}
