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
