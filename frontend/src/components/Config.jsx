import { useState, useEffect, useCallback } from 'react'
import { getConfig, addProfile, deleteProfile, activateProfile, testConnection, detectEnvCreds, listAwsProfiles, listS3Buckets, createS3Bucket, selectS3Bucket } from '../api/client'
import { Card, SectionLabel } from './ui/Section'

const PROVIDER_ORDER = ['aws', 'azure', 'gcp', 'huggingface', 'elevenlabs', 'openai', 'anthropic', 'replicate', 'deepgram']

export default function Config() {
  const [providers, setProviders] = useState({})
  const [error, setError] = useState('')

  const reload = useCallback(() => {
    getConfig().then(setProviders).catch(e => setError(e.message))
  }, [])

  useEffect(reload, [reload])

  const sorted = PROVIDER_ORDER.filter(k => providers[k]).map(k => [k, providers[k]])

  return (
    <div className="max-w-3xl space-y-5">
      {error && <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>}
      {sorted.map(([pid, pdata]) => (
        <ProviderCard key={pid} pid={pid} data={pdata} onRefresh={reload} />
      ))}
    </div>
  )
}

function ProviderCard({ pid, data, onRefresh }) {
  const { label, auth_methods = [], items, active_id } = data
  const [adding, setAdding] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  const activeItem = items.find(it => it.id === active_id)
  const activeBucket = activeItem?.fields?.AWS_TRANSCRIBE_BUCKET || ''

  const handleDelete = async (profId) => {
    if (!confirm('Delete this profile?')) return
    setBusy(true); setMsg('')
    try { await deleteProfile(pid, profId); onRefresh() }
    catch (e) { setMsg(e.message) }
    finally { setBusy(false) }
  }

  const handleActivate = async (profId) => {
    setBusy(true); setMsg('')
    try { await activateProfile(pid, profId); onRefresh() }
    catch (e) { setMsg(e.message) }
    finally { setBusy(false) }
  }

  const handleAdded = () => { setAdding(false); onRefresh() }

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <SectionLabel>{label}</SectionLabel>
          {items.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-500">{items.length} profile{items.length !== 1 ? 's' : ''}</span>
          )}
        </div>
        {!adding && (
          <button onClick={() => setAdding(true)}
            className="text-xs px-3 py-1 bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] rounded-lg hover:bg-[var(--accent)]/20 transition">
            + Add credentials
          </button>
        )}
      </div>

      {items.length === 0 && !adding && (
        <p className="text-xs text-slate-500 italic">No credentials stored. Click "+ Add credentials" to add one.</p>
      )}

      <div className="space-y-2">
        {items.map(item => (
          <ProfileChip key={item.id} pid={pid} item={item} isActive={item.id === active_id}
            authMethods={auth_methods} busy={busy}
            onActivate={() => handleActivate(item.id)}
            onDelete={() => handleDelete(item.id)} />
        ))}
      </div>

      {pid === 'aws' && items.length > 0 && active_id && (
        <BucketPicker currentBucket={activeBucket} onRefresh={onRefresh} />
      )}

      {adding && (
        <AddForm authMethods={auth_methods} pid={pid} onDone={handleAdded} onCancel={() => setAdding(false)} />
      )}

      {msg && <p className="text-xs text-red-400 mt-2">{msg}</p>}
    </Card>
  )
}

function ProfileChip({ pid, item, isActive, authMethods, busy, onActivate, onDelete }) {
  const [expanded, setExpanded] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [visibleSecrets, setVisibleSecrets] = useState({})

  const method = authMethods.find(m => m.id === item.auth_method) || authMethods[0] || {}
  const fieldDefs = method.fields || {}

  const toggleSecretVisibility = (fk) => {
    setVisibleSecrets(prev => ({ ...prev, [fk]: !prev[fk] }))
  }

  const handleTest = async () => {
    setTesting(true); setTestResult(null)
    try { setTestResult(await testConnection(pid, item.id)) }
    catch (e) { setTestResult({ ok: false, error: e.message }) }
    finally { setTesting(false) }
  }

  return (
    <div className={`rounded-lg border transition-colors ${isActive ? 'bg-[var(--accent)]/5 border-[var(--accent)]/40' : 'bg-slate-800/40 border-slate-700'}`}>
      <div className="flex items-center gap-2 px-3 py-2">
        <button onClick={onActivate} disabled={busy || isActive} title={isActive ? 'Active' : 'Click to activate'}
          className={`w-4 h-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 transition-colors ${isActive ? 'border-[var(--accent)] bg-[var(--accent)]' : 'border-slate-500 hover:border-[var(--accent)]'}`}>
          {isActive && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
        </button>

        <span className={`text-sm font-medium flex-1 truncate ${isActive ? 'text-[var(--accent)]' : 'text-slate-300'}`}>{item.label}</span>

        <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/80 text-slate-400 font-medium">{method.label || item.auth_method}</span>

        {isActive && <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[var(--accent)]/20 text-[var(--accent)]">ACTIVE</span>}

        <button onClick={handleTest} disabled={testing} title="Test connection"
          className="text-slate-500 hover:text-green-400 transition-colors p-0.5 disabled:opacity-40">
          {testing
            ? <svg className="w-3.5 h-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
            : <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
        </button>

        <button onClick={() => setExpanded(e => !e)} className="text-slate-500 hover:text-slate-300 transition-colors p-0.5">
          <svg className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" /></svg>
        </button>

        <button onClick={onDelete} disabled={busy} title="Delete"
          className="text-slate-500 hover:text-red-400 transition-colors p-0.5 disabled:opacity-40">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
        </button>
      </div>

      {testResult && (
        <div className={`mx-3 mb-2 px-2.5 py-1.5 rounded text-[11px] flex items-center gap-1.5 ${testResult.ok ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
          {testResult.ok
            ? <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>
            : <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>}
          <span className="truncate">{testResult.ok ? testResult.message : (testResult.error || 'Connection failed')}</span>
        </div>
      )}

      {expanded && (
        <div className="px-3 pb-2 pt-1 border-t border-slate-700/50">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {Object.entries(fieldDefs).map(([fk, fdef]) => {
              const val = item.fields[fk]
              if (!val) return null
              const isSecret = !!fdef.secret
              const isRevealed = visibleSecrets[fk]
              return (
                <div key={fk} className="flex items-center gap-1.5">
                  <span className="text-[10px] text-slate-500">{fdef.label}:</span>
                  <span className="text-[10px] text-slate-300 font-mono truncate">
                    {isSecret && !isRevealed ? '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022' : val}
                  </span>
                  {isSecret && (
                    <button onClick={() => toggleSecretVisibility(fk)}
                      className="text-[9px] text-slate-500 hover:text-slate-300 transition ml-0.5 flex-shrink-0">
                      {isRevealed ? 'Hide' : 'Show'}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
          {item.auth_method === 'env' && <p className="text-[10px] text-slate-500 italic mt-1">Using environment variables</p>}
        </div>
      )}
    </div>
  )
}

function AddForm({ authMethods, pid, onDone, onCancel }) {
  const [label, setLabel] = useState('')
  const [selectedMethod, setSelectedMethod] = useState(authMethods[0]?.id || '')
  const [fields, setFields] = useState({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [envStatus, setEnvStatus] = useState(null)
  const [awsProfiles, setAwsProfiles] = useState(null)

  const method = authMethods.find(m => m.id === selectedMethod) || authMethods[0] || {}
  const fieldDefs = method.fields || {}
  const isEnv = selectedMethod === 'env' || selectedMethod === 'adc'

  useEffect(() => {
    setFields({})
    setError('')
    setEnvStatus(null)
    setAwsProfiles(null)
    const defaults = {}
    for (const [fk, fdef] of Object.entries(method.fields || {})) {
      if (fdef.default) defaults[fk] = fdef.default
    }
    setFields(defaults)

    if (isEnv) {
      detectEnvCreds(pid).then(setEnvStatus).catch(() => {})
    }
    if (selectedMethod === 'profile' && pid === 'aws') {
      listAwsProfiles().then(d => setAwsProfiles(d.profiles || [])).catch(() => setAwsProfiles([]))
    }
  }, [selectedMethod, pid, isEnv, method])

  const set = key => e => setFields(prev => ({ ...prev, [key]: e.target.value }))
  const fieldEntries = Object.entries(fieldDefs)
  const hasRequired = fieldEntries.filter(([, fd]) => fd.required).every(([fk]) => fields[fk]?.trim())
  const canSave = isEnv || hasRequired

  const handleSave = async () => {
    if (!canSave) { setError('Fill in all required fields'); return }
    setSaving(true); setError('')
    try {
      await addProfile(pid, label || '', fields, selectedMethod)
      onDone()
    } catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  return (
    <div className="mt-3 pt-3 border-t border-slate-700/50 space-y-3">
      {authMethods.length > 1 && (
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1.5 block">Auth Method</span>
          <div className="flex flex-wrap gap-1.5">
            {authMethods.map(m => (
              <button key={m.id} onClick={() => setSelectedMethod(m.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${selectedMethod === m.id
                  ? 'bg-[var(--accent)] text-slate-900'
                  : 'bg-slate-700/60 text-slate-400 hover:bg-slate-700 hover:text-slate-200'}`}>
                {m.label}
              </button>
            ))}
          </div>
          {method.description && <p className="text-[10px] text-slate-500 mt-1.5">{method.description}</p>}
        </div>
      )}

      {isEnv && envStatus && (
        <div className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border ${envStatus.detected ? 'bg-green-500/10 border-green-500/20' : 'bg-amber-500/10 border-amber-500/20'}`}>
          {envStatus.detected
            ? <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            : <svg className="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" /></svg>}
          <div className="flex-1">
            <span className={`text-xs font-medium ${envStatus.detected ? 'text-green-300' : 'text-amber-300'}`}>
              {envStatus.detected ? 'Environment variables detected' : 'Some variables not found'}
            </span>
            <div className="flex flex-wrap gap-2 mt-1">
              {Object.entries(envStatus.keys || {}).map(([k, found]) => (
                <span key={k} className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${found ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'}`}>
                  {found ? '✓' : '✗'} {k}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {!isEnv && (
        <>
          <div>
            <span className="text-xs text-slate-400">Profile name <span className="text-slate-600">(optional)</span></span>
            <input type="text" value={label} onChange={e => setLabel(e.target.value)} placeholder="e.g. Production, Personal"
              className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-500 mt-1" />
          </div>

          {selectedMethod === 'profile' && pid === 'aws' && awsProfiles !== null && (
            <div>
              <span className="text-xs text-slate-400">Select from ~/.aws/credentials</span>
              {awsProfiles.length === 0 ? (
                <p className="text-[10px] text-slate-500 mt-1">No profiles found in ~/.aws/credentials</p>
              ) : (
                <div className="flex flex-wrap gap-1.5 mt-1.5">
                  {awsProfiles.map(p => (
                    <button key={p} onClick={() => setFields(prev => ({ ...prev, AWS_PROFILE_NAME: p }))}
                      className={`px-2.5 py-1 rounded text-xs font-mono transition ${fields.AWS_PROFILE_NAME === p
                        ? 'bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40'
                        : 'bg-slate-700/60 text-slate-400 border border-slate-600 hover:text-slate-200'}`}>
                      {p}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {fieldEntries.map(([fk, fdef]) => {
              if (selectedMethod === 'profile' && pid === 'aws' && fk === 'AWS_PROFILE_NAME' && awsProfiles?.length > 0) return null
              return (
                <div key={fk} className={fdef.multiline ? 'md:col-span-2' : ''}>
                  <span className="text-xs text-slate-400">
                    {fdef.label}
                    {!fdef.required && <span className="text-slate-600 ml-1">(optional)</span>}
                  </span>
                  {fdef.multiline ? (
                    <textarea value={fields[fk] || ''} onChange={set(fk)} rows={6} placeholder="Paste JSON content..."
                      className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-500 mt-1 font-mono resize-y" />
                  ) : (
                    <input type={fdef.secret ? 'password' : 'text'} value={fields[fk] || ''} onChange={set(fk)}
                      placeholder={fdef.secret ? '••••••' : (fdef.default || '')}
                      className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-500 mt-1" />
                  )}
                </div>
              )
            })}
          </div>
        </>
      )}

      {isEnv && (
        <div>
          <span className="text-xs text-slate-400">Profile name <span className="text-slate-600">(optional)</span></span>
          <input type="text" value={label} onChange={e => setLabel(e.target.value)} placeholder="e.g. From Environment"
            className="w-full bg-slate-700/80 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-500 mt-1" />
        </div>
      )}

      {error && <p className="text-xs text-red-400">{error}</p>}
      <div className="flex gap-2">
        <button onClick={handleSave} disabled={saving || !canSave}
          className="px-5 py-2 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-bold rounded-lg disabled:opacity-40 text-sm transition">
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button onClick={onCancel} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm transition">Cancel</button>
      </div>
    </div>
  )
}

function BucketPicker({ currentBucket, onRefresh }) {
  const [buckets, setBuckets] = useState([])
  const [loading, setLoading] = useState(false)
  const [creating, setCreating] = useState(false)
  const [selecting, setSelecting] = useState(null)
  const [error, setError] = useState('')
  const [loaded, setLoaded] = useState(false)

  const load = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const d = await listS3Buckets()
      setBuckets(d.buckets || [])
      setLoaded(true)
    } catch (e) {
      setError(e.message?.includes('credentials') ? 'AWS credentials may be invalid' : (e.message || 'Failed'))
    } finally { setLoading(false) }
  }, [])

  const handleSelect = async (name) => {
    setSelecting(name); setError('')
    try {
      await selectS3Bucket(name)
      onRefresh()
    } catch (e) { setError(e.message) }
    finally { setSelecting(null) }
  }

  const handleCreate = async () => {
    setCreating(true); setError('')
    try {
      const d = await createS3Bucket()
      setBuckets(prev => [...prev, d.bucket?.Name || d.bucket || ''])
      onRefresh()
    } catch (e) { setError(e.message) }
    finally { setCreating(false) }
  }

  return (
    <div className="mt-3 pt-3 border-t border-slate-700/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-400">Transcribe S3 Bucket</span>
        <div className="flex items-center gap-2">
          <button onClick={load} disabled={loading} title="Load bucket list"
            className="text-slate-500 hover:text-[var(--accent)] disabled:opacity-40 transition-colors">
            <svg className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <button onClick={handleCreate} disabled={creating}
            className="text-[10px] px-2.5 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 disabled:opacity-40 transition">
            {creating ? 'Creating…' : '+ Create new bucket'}
          </button>
        </div>
      </div>

      {/* Warning if no bucket selected */}
      {!currentBucket && (
        <div className="flex items-center gap-2 px-3 py-2 bg-amber-500/10 border border-amber-500/20 rounded-lg mb-2">
          <svg className="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <span className="text-xs text-amber-300">No S3 bucket selected. Audio transfer verification (AWS Transcribe) requires a bucket. Click the refresh icon to load your buckets, or create a new one.</span>
        </div>
      )}

      {/* Currently selected */}
      {currentBucket && (
        <div className="flex items-center gap-2 px-3 py-2 bg-green-500/10 border border-green-500/20 rounded-lg mb-2">
          <span className="w-2 h-2 bg-green-400 rounded-full flex-shrink-0" />
          <span className="text-xs text-green-300">Active: <span className="font-mono font-medium">{currentBucket}</span></span>
        </div>
      )}

      {/* Bucket list */}
      {loaded && buckets.length > 0 && (
        <div className="space-y-1">
          {buckets.map(b => {
            const isActive = b === currentBucket
            const isSelecting = selecting === b
            return (
              <button key={b} onClick={() => !isActive && handleSelect(b)} disabled={isActive || !!selecting}
                className={`w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-colors ${
                  isActive
                    ? 'bg-[var(--accent)]/10 border border-[var(--accent)]/40 text-[var(--accent)]'
                    : 'bg-slate-800/40 border border-slate-700 text-slate-300 hover:border-[var(--accent)]/40 hover:text-[var(--accent)]'
                } disabled:opacity-60`}>
                <span className={`w-3 h-3 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${isActive ? 'border-[var(--accent)] bg-[var(--accent)]' : 'border-slate-500'}`}>
                  {isActive && <span className="w-1.5 h-1.5 bg-white rounded-full" />}
                </span>
                <span className="flex-1">{b}</span>
                {isSelecting && <span className="text-[10px] text-slate-500">saving…</span>}
                {isActive && <span className="text-[10px] font-semibold text-[var(--accent)]">ACTIVE</span>}
              </button>
            )
          })}
        </div>
      )}

      {loaded && buckets.length === 0 && (
        <p className="text-[10px] text-slate-500">No buckets found. Create one to use AWS Transcribe.</p>
      )}

      {!loaded && !loading && (
        <p className="text-[10px] text-slate-500">Click the refresh icon to load available buckets.</p>
      )}

      {error && <p className="text-[10px] text-red-400 mt-1">{error}</p>}
    </div>
  )
}
