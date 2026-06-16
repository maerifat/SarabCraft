import { createContext, useContext, useState, useRef } from 'react'

const ImageAttackContext = createContext(null)

/**
 * Provides persistent image-attack state that survives React-Router
 * navigation.  Only the "important" fields that the user would lose
 * on a tab switch live here; ephemeral UI bits (searchTerm,
 * transferModal, infoOpen) stay local to ImageAttackTab.
 */
export function ImageAttackProvider({ children }) {
  const [result, setResult]             = useState(null)
  const [inputFile, setInputFile]       = useState(null)
  const [inputPreview, setInputPreview] = useState(null)
  const [targetFile, setTargetFile]     = useState(null)
  const [targetPreview, setTargetPreview] = useState(null)
  const [model, setModel]               = useState('')
  const [attack, setAttack]             = useState('PGD')
  const [paramValues, setParamValues]   = useState({})
  const [ensembleModels, setEnsembleModels] = useState([])
  const [ensembleMode, setEnsembleMode] = useState('Simultaneous')
  const [loading, setLoading]           = useState(false)
  const [error, setError]               = useState('')
  const [currentJobId, setCurrentJobId] = useState('')

  /* refs that ImageAttackTab also needs across mounts */
  const abortRef   = useRef(null)
  const jobIdRef   = useRef('')

  /**
   * Rehydrate the attack workspace from a persisted job so the user can
   * recover transfer testing, report export, metrics and images after a
   * page refresh or by reopening a completed job from the Jobs page.
   */
  const restoreFromJob = (job, { inputDataUrl = null } = {}) => {
    const fields = job?.request?.fields || {}
    const res = job?.result || null

    if (fields.attack) setAttack(fields.attack)
    if (fields.model) setModel(fields.model)

    const RESERVED = new Set(['attack', 'model', 'ensemble_mode', 'ensemble_models', 'ensemble_model_snapshots_json'])
    const restoredParams = {}
    Object.entries(fields).forEach(([key, value]) => {
      if (RESERVED.has(key)) return
      const num = typeof value === 'string' && value.trim() !== '' && !Number.isNaN(Number(value)) ? Number(value) : value
      if (value === 'true') restoredParams[key] = true
      else if (value === 'false') restoredParams[key] = false
      else restoredParams[key] = num
    })
    setParamValues(restoredParams)

    if (fields.ensemble_mode) setEnsembleMode(fields.ensemble_mode)
    if (fields.ensemble_models) {
      const list = String(fields.ensemble_models).split(',').map(s => s.trim()).filter(Boolean)
      setEnsembleModels(list)
    } else {
      setEnsembleModels([])
    }

    setResult(res)
    setCurrentJobId(job?.job_id || '')
    setLoading(false)
    setError('')

    if (inputDataUrl) {
      setInputPreview(inputDataUrl)
      setInputFile(null)
    } else {
      setInputPreview(null)
      setInputFile(null)
    }
    setTargetPreview(null)
    setTargetFile(null)
  }

  return (
    <ImageAttackContext.Provider value={{
      result, setResult,
      inputFile, setInputFile,
      inputPreview, setInputPreview,
      targetFile, setTargetFile,
      targetPreview, setTargetPreview,
      model, setModel,
      attack, setAttack,
      paramValues, setParamValues,
      ensembleModels, setEnsembleModels,
      ensembleMode, setEnsembleMode,
      loading, setLoading,
      error, setError,
      currentJobId, setCurrentJobId,
      abortRef,
      jobIdRef,
      restoreFromJob,
    }}>
      {children}
    </ImageAttackContext.Provider>
  )
}

export function useImageAttack() {
  const ctx = useContext(ImageAttackContext)
  if (!ctx) throw new Error('useImageAttack must be used inside <ImageAttackProvider>')
  return ctx
}
