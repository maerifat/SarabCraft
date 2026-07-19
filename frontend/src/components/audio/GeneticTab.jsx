import { useState, useEffect } from 'react'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, TextInput, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, SimpleAsrResult, PreviewTranscription, fileToBase64 } from './AudioShared'

export default function GeneticTab({ asrKeys, loading, error, result, onRun, onTransfer }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (asrKeys.length && !model) setModel(asrKeys[0]?.value || asrKeys[0]) }, [asrKeys, model])
  const [targetText, setTargetText] = useState('')
  const [eps, setEps] = useState(0.05)
  const [pop, setPop] = useState(20)
  const [genIter, setGenIter] = useState(300)
  const [geIter, setGeIter] = useState(100)

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const go = async () => {
    if (!file || !targetText) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('target_text', targetText)
    fd.append('epsilon', eps); fd.append('population_size', pop)
    fd.append('genetic_iterations', genIter); fd.append('gradient_estimation_iterations', geIter)
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
        <Slider label="Epsilon (L∞)" value={eps} onChange={setEps} min={0.01} max={0.3} step={0.01} defaultValue={0.05} />
        <Slider label="Population size" value={pop} onChange={setPop} min={5} max={60} step={5} defaultValue={20} />
        <Slider label="Genetic iterations" value={genIter} onChange={setGenIter} min={50} max={1000} step={50} defaultValue={300} />
        <Slider label="Gradient-est. iterations" value={geIter} onChange={setGeIter} min={0} max={300} step={10} defaultValue={100} />
      </ParamGrid>
      <p className="text-[11px] text-slate-500 mt-2">Black-box: only the transcription output is used (no gradients). Phase 1 drives a genetic search to match the target; phase 2 uses NES gradient estimation to shrink the perturbation.</p>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Genetic Attack" /><ErrorMsg msg={error} /></div>
    {result && <SimpleAsrResult result={result} originalWavB64={origB64} onTransfer={() => onTransfer?.({ originalWavB64: origB64, targetText })} />}
  </>)
}
