import { useState, useEffect } from 'react'
import { getAudioLabels } from '../../api/client'
import Slider from '../ui/Slider'
import { Card, SectionLabel, InputGrid, ParamGrid, Select, RunButton, ErrorMsg } from '../ui/Section'
import { AudioInput, AudioPlayer, fileToBase64 } from './AudioShared'

export default function ClassificationTab({ models: audioModels, attacks, loading, error, result, onRun }) {
  const [file, setFile] = useState(null)
  const [origB64, setOrigB64] = useState(null)
  const [model, setModel] = useState('')
  useEffect(() => { if (audioModels.length && !model) setModel(audioModels[0]?.value || audioModels[0]) }, [audioModels, model])
  const [atk, setAtk] = useState('PGD')
  const [labels, setLabels] = useState([])
  const [tgt, setTgt] = useState('')
  const [eps, setEps] = useState(0.02)
  const [iter, setIter] = useState(50)
  const [alpha, setAlpha] = useState(1.0)
  const [mom, setMom] = useState(1.0)
  const [os2, setOs2] = useState(0.02)
  const [cwK, setCwK] = useState(0)
  const [cwLr, setCwLr] = useState(0.01)
  const [cwC, setCwC] = useState(1.0)

  useEffect(() => {
    if (model) getAudioLabels(model).then(d => { setLabels(d.labels || []); setTgt(d.labels?.[0] || '') }).catch(() => setLabels([]))
  }, [model])

  useEffect(() => {
    if (!file) { setOrigB64(null); return }
    fileToBase64(file).then(setOrigB64).catch(() => setOrigB64(null))
  }, [file])

  const atkList = attacks?.length ? attacks : ['FGSM', 'I-FGSM (BIM)', 'PGD', 'MI-FGSM', 'DeepFool', 'C&W (L2)']

  const go = async () => {
    if (!file || !tgt) return
    const fd = new FormData()
    fd.append('audio_file', file); fd.append('model', model); fd.append('target_label', tgt); fd.append('attack', atk)
    fd.append('epsilon', String(eps)); fd.append('iterations', String(iter)); fd.append('alpha', String(alpha))
    fd.append('momentum_decay', String(mom)); fd.append('overshoot', String(os2))
    fd.append('cw_confidence', String(cwK)); fd.append('cw_lr', String(cwLr)); fd.append('cw_c', String(cwC))
    await onRun(fd)
  }

  return (<>
    <Card><SectionLabel>Input</SectionLabel>
      <InputGrid>
        <AudioInput label="Audio file" file={file} setFile={setFile} />
        <Select label="Model" value={model} onChange={setModel} options={audioModels} />
        <Select label="Attack" value={atk} onChange={setAtk} options={atkList} />
        <Select label="Target label" value={tgt} onChange={setTgt} options={labels} />
      </InputGrid>
    </Card>
    <Card><SectionLabel>Parameters</SectionLabel>
      <ParamGrid>
        <Slider label="Epsilon" value={eps} onChange={setEps} min={0.001} max={0.2} step={0.001} defaultValue={0.02} />
        <Slider label="Iterations" value={iter} onChange={setIter} min={1} max={500} step={1} defaultValue={50} />
        {!['FGSM', 'DeepFool', 'C&W (L2)'].includes(atk) && <Slider label="Alpha" value={alpha} onChange={setAlpha} min={0.1} max={10} step={0.1} defaultValue={1.0} />}
        {atk === 'MI-FGSM' && <Slider label="Momentum (μ)" value={mom} onChange={setMom} min={0} max={1} step={0.1} defaultValue={1.0} />}
        {atk === 'DeepFool' && <Slider label="Overshoot" value={os2} onChange={setOs2} min={0.01} max={0.5} step={0.01} defaultValue={0.02} />}
        {atk === 'C&W (L2)' && <>
          <Slider label="C&W κ" value={cwK} onChange={setCwK} min={0} max={50} step={1} defaultValue={0} />
          <Slider label="C&W LR" value={cwLr} onChange={setCwLr} min={0.001} max={0.1} step={0.001} defaultValue={0.01} />
          <Slider label="C&W c" value={cwC} onChange={setCwC} min={0.1} max={100} step={0.1} defaultValue={1.0} />
        </>}
      </ParamGrid>
    </Card>
    <div className="flex items-center gap-4"><RunButton onClick={go} loading={loading} label="Run Audio Attack" /><ErrorMsg msg={error} /></div>
    {result && (
      <Card className="border-[var(--accent)]/30"><SectionLabel>Results</SectionLabel>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <p className={`text-lg font-bold ${result.success ? 'text-green-400' : 'text-amber-400'}`}>{result.success ? 'Target reached' : 'Partial — try higher ε'}</p>
            <p className="text-sm text-slate-300 mt-2">Original: <span className="text-slate-400">{result.original_class}</span> → Adversarial: <span className="font-medium">{result.adversarial_class}</span></p>
          </div>
          <div className="space-y-3">
            {origB64 && <AudioPlayer wavB64={origB64} label="Original audio" filename="original_audio.wav" />}
            <AudioPlayer wavB64={result.adversarial_wav_b64} label="Adversarial audio" />
          </div>
        </div>
      </Card>
    )}
  </>)
}
