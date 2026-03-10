import { useState, useEffect, useMemo } from 'react'
import {
  getAsrModels, getAudioModels, getAudioAttacks,
  runTranscriptionAttack, runHiddenCommandAttack, runUniversalMutingAttack,
  runPsychoacousticAttack, runOverTheAirAttack, runSpeechJammingAttack,
  runAudioClassificationAttack, getUA3Models, runUA3Attack,
} from '../api/client'
import { Card, SectionLabel } from './ui/Section'
import AudioTransferModal from './AudioTransferModal'
import AudioAttackSelector from './audio/AudioAttackSelector'
import TranscriptionTab from './audio/TranscriptionTab'
import HiddenCommandTab from './audio/HiddenCommandTab'
import MutingTab from './audio/MutingTab'
import PsychoacousticTab from './audio/PsychoacousticTab'
import OTATab from './audio/OTATab'
import JammingTab from './audio/JammingTab'
import UA3Tab from './audio/UA3Tab'
import ClassificationTab from './audio/ClassificationTab'
import { AUDIO_ATTACK_REGISTRY, THREAT_COLORS, THREAT_LABEL } from './audio/audioAttackRegistry'

function normalizeModelItems(response) {
  if (Array.isArray(response?.items)) return response.items.map(item => ({
    label: item.label || item.display_name || item.value || item.id,
    value: item.value || item.id || item.model_ref,
  })).filter(item => item.value)
  return (response?.models || []).map(value => ({ label: value, value }))
}

export default function AudioTabs() {
  const [subTab, setSubTab] = useState('transcription')
  const [asrModels, setAsrModels] = useState([])
  const [audioModels, setAudioModels] = useState([])
  const [audioAttacksList, setAudioAttacksList] = useState([])
  const [ua3ModelsList, setUa3ModelsList] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [transferModal, setTransferModal] = useState(false)
  const [transferContext, setTransferContext] = useState(null)

  useEffect(() => {
    Promise.allSettled([getAsrModels(), getAudioModels(), getAudioAttacks(), getUA3Models()]).then(results => {
      const [a, m, att, ua3] = results
      if (a.status === 'fulfilled') setAsrModels(normalizeModelItems(a.value))
      if (m.status === 'fulfilled') setAudioModels(normalizeModelItems(m.value))
      if (att.status === 'fulfilled') setAudioAttacksList(att.value.attacks || [])
      if (ua3.status === 'fulfilled') setUa3ModelsList(ua3.value.models || [])
    })
  }, [])

  const asrKeys = useMemo(() => asrModels, [asrModels])
  const audioModelKeys = useMemo(() => audioModels, [audioModels])

  const run = fn => async fd => {
    setLoading(true)
    setError('')
    setResult(null)
    setTransferContext(null)
    try {
      setResult(await fn(fd))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }
  const openTransfer = context => {
    setTransferContext(context || null)
    setTransferModal(true)
  }

  const shared = { loading, error, result }
  const meta = AUDIO_ATTACK_REGISTRY[subTab]

  const handleSelect = id => {
    setSubTab(id)
    setResult(null)
    setError('')
  }

  return (
    <div className="space-y-4">
      <Card>
        <SectionLabel>Attack Method</SectionLabel>
        <AudioAttackSelector selected={subTab} onSelect={handleSelect} />
      </Card>

      {meta && (
        <div className="flex items-center gap-3 px-1">
          <span className="text-sm font-semibold text-slate-200">{meta.name}</span>
          <span className={`text-[9px] px-1.5 py-0.5 rounded-full border font-medium ${THREAT_COLORS[meta.threat]}`}>
            {THREAT_LABEL[meta.threat]}
          </span>
          <span className="text-[10px] text-slate-500">{meta.authors} · {meta.year} · {meta.norm}</span>
        </div>
      )}

      {subTab === 'transcription' && <TranscriptionTab asrKeys={asrKeys} {...shared} onRun={run(runTranscriptionAttack)} onTransfer={openTransfer} />}
      {subTab === 'hidden' && <HiddenCommandTab asrKeys={asrKeys} {...shared} onRun={run(runHiddenCommandAttack)} onTransfer={openTransfer} />}
      {subTab === 'muting' && <MutingTab asrKeys={asrKeys} {...shared} onRun={run(runUniversalMutingAttack)} onTransfer={openTransfer} />}
      {subTab === 'psycho' && <PsychoacousticTab asrKeys={asrKeys} {...shared} onRun={run(runPsychoacousticAttack)} onTransfer={openTransfer} />}
      {subTab === 'ota' && <OTATab asrKeys={asrKeys} {...shared} onRun={run(runOverTheAirAttack)} onTransfer={openTransfer} />}
      {subTab === 'jamming' && <JammingTab asrKeys={asrKeys} {...shared} onRun={run(runSpeechJammingAttack)} onTransfer={openTransfer} />}
      {subTab === 'ua3' && <UA3Tab models={ua3ModelsList} asrKeys={asrKeys} {...shared} onRun={run(runUA3Attack)} onTransfer={openTransfer} />}
      {subTab === 'classification' && <ClassificationTab models={audioModelKeys} attacks={audioAttacksList} {...shared} onRun={run(runAudioClassificationAttack)} />}

      {transferModal && result?.adversarial_wav_b64 && (
        <AudioTransferModal
          adversarialWavB64={result.adversarial_wav_b64}
          originalWavB64={transferContext?.originalWavB64}
          sampleRate={result.sample_rate || 16000}
          targetText={transferContext?.targetText ?? result.target_text ?? result.command_text ?? ''}
          originalText={result.original_text}
          onClose={() => setTransferModal(false)}
        />
      )}
    </div>
  )
}
