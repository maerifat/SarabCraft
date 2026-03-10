import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function MutingTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [mode, setMode] = useState('Mute (Silence)')
  const [targetText, setTargetText] = useState('')
  const [seg, setSeg] = useState(0.64)
  const [iter, setIter] = useState(300)
  const [lr, setLr] = useState(0.01)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('mode', mode)
    fd.append('target_text', targetText); fd.append('segment_duration', seg)
    fd.append('iterations', iter); fd.append('lr', lr)
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="ASR Model" value={model} onChange={setModel} options={asrKeys} />
        <Select label="Mode" value={mode} onChange={setMode} options={['Mute (Silence)', 'Targeted Override']} />
      </InputGrid>
      {mode === 'Targeted Override' && (
        <div className="mt-3 max-w-md">
          <TextInput label="Override text" value={targetText} onChange={setTargetText} placeholder="Target text" />
        </div>
      )}
      <div className="mt-3"><PreviewTranscription file={file} model={model} /></div>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Segment (s)" value={seg} onChange={setSeg} min={0.3} max={2} step={0.01} defaultValue={0.64} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={50} max={2000} step={50} defaultValue={300} />
        <Slider label="Learning rate" value={lr} onChange={setLr} min={0.001} max={0.05} step={0.001} defaultValue={0.01} />
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Learn Universal Segment" /><ErrorMsg msg={error} /></div>
    {result && (
      <SimpleAsrResult
        result={result}
        originalWavB64={origB64}
        onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText: mode === 'Targeted Override' ? targetText : '' })}
      />
    )}
  </>)
}
