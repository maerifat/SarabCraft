import { createContext, useContext, useState, useRef } from 'react'

const TextAttackContext = createContext(null)

/**
 * Provides persistent text-attack state that survives React-Router
 * navigation.  Mirrors ImageAttackProvider — keeps "important" fields
 * (input text, selected model/attack, params, result, job state) in
 * context so they persist when the user switches to Jobs and back.
 */
export function TextAttackProvider({ children }) {
  const [model, setModel]             = useState('')
  const [attack, setAttack]           = useState('')
  const [text, setText]               = useState('')
  const [targetLabel, setTargetLabel] = useState('')
  const [params, setParams]           = useState({})
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState('')
  const [result, setResult]           = useState(null)
  const [jobId, setJobId]             = useState('')

  /* refs that TextAttackTab also needs across mounts */
  const abortRef = useRef(null)

  return (
    <TextAttackContext.Provider value={{
      model, setModel,
      attack, setAttack,
      text, setText,
      targetLabel, setTargetLabel,
      params, setParams,
      loading, setLoading,
      error, setError,
      result, setResult,
      jobId, setJobId,
      abortRef,
    }}>
      {children}
    </TextAttackContext.Provider>
  )
}

export function useTextAttack() {
  const ctx = useContext(TextAttackContext)
  if (!ctx) throw new Error('useTextAttack must be used inside <TextAttackProvider>')
  return ctx
}
