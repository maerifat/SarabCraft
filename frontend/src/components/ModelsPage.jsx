import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  createModelEntry,
  deleteModelEntry,
  duplicateModelEntry,
  getConfig,
  getModelCatalog,
  testModelEntry,
  toggleModelEntry,
  updateModelEntry,
} from '../api/client'
import { Card, SectionLabel } from './ui/Section'

const TAB_OPTIONS = [
  { id: 'image', label: 'Image' },
  { id: 'audio', label: 'Audio' },
]

const SERVICE_OPTIONS = {
  image: ['AWS Rekognition', 'Azure Vision', 'GCP Vision'],
  audio: ['AWS Transcribe', 'ElevenLabs STT'],
}

const PROVIDER_HINTS = {
  aws: 'AWS credentials are managed in Settings > Credentials.',
  azure: 'Azure credentials are managed in Settings > Credentials.',
  gcp: 'GCP credentials are managed in Settings > Credentials.',
  huggingface: 'Hugging Face credentials are optional for some targets, but storing a linked profile keeps the setup explicit.',
  elevenlabs: 'ElevenLabs credentials are managed in Settings > Credentials.',
}

const INPUT_CLASS = 'w-full bg-slate-800/70 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:border-[var(--accent)] outline-none transition'
const ACTION_BUTTON_CLASS = 'px-3 py-1.5 rounded-lg text-[11px] font-medium bg-slate-800/50 border border-slate-700 text-slate-300 hover:text-[var(--accent)] transition disabled:opacity-50'
const ACTION_BUTTON_DANGER_CLASS = 'px-3 py-1.5 rounded-lg text-[11px] font-medium bg-red-500/10 border border-red-500/20 text-red-300 hover:bg-red-500/20 transition disabled:opacity-50'

const DEFAULTS = {
  image_source: {
    kind: 'source_model',
    domain: 'image',
    task: 'image_classification',
    backend: 'hf_local_image',
    display_name: '',
    description: '',
    provider: 'Hugging Face',
    family: 'Custom',
    model_ref: '',
    settings: { trust_remote_code: false },
    compatibility: ['classify', 'attack_source', 'robustness_source', 'benchmark_source', 'verification_target_local', 'benchmark_target_local'],
    aliases: [],
    item_ids: [],
    credential_profile_id: '',
    enabled: true,
    sort_order: 0,
  },
  audio_source: {
    kind: 'source_model',
    domain: 'audio',
    task: 'audio_classification',
    backend: 'hf_local_audio',
    display_name: '',
    description: '',
    provider: 'Hugging Face',
    family: 'Custom',
    model_ref: '',
    settings: { trust_remote_code: false },
    compatibility: ['attack_source'],
    aliases: [],
    item_ids: [],
    credential_profile_id: '',
    enabled: true,
    sort_order: 0,
  },
  asr_source: {
    kind: 'source_model',
    domain: 'audio',
    task: 'asr',
    backend: 'hf_local_asr',
    display_name: '',
    description: '',
    provider: 'Hugging Face',
    family: 'ASR',
    model_ref: '',
    settings: { trust_remote_code: false },
    compatibility: ['attack_source', 'robustness_source', 'benchmark_source'],
    aliases: [],
    item_ids: [],
    credential_profile_id: '',
    enabled: true,
    sort_order: 0,
  },
  hf_target: {
    kind: 'verification_target',
    domain: 'image',
    task: 'image_verification',
    backend: 'hf_api',
    display_name: '',
    description: '',
    provider: 'huggingface',
    family: 'HuggingFace API',
    model_ref: '',
    settings: { service_name: 'HuggingFace API' },
    compatibility: ['verification_target', 'benchmark_target'],
    aliases: [],
    item_ids: [],
    credential_profile_id: '',
    enabled: true,
    sort_order: 0,
  },
  image_service_target: {
    kind: 'verification_target',
    domain: 'image',
    task: 'image_verification',
    backend: 'verifier_service',
    display_name: '',
    description: '',
    provider: 'aws',
    family: 'Cloud Image',
    model_ref: 'AWS Rekognition',
    settings: { service_name: 'AWS Rekognition', provider: 'aws' },
    compatibility: ['verification_target', 'benchmark_target'],
    aliases: [],
    item_ids: [],
    credential_profile_id: '',
    enabled: true,
    sort_order: 0,
  },
  audio_service_target: {
    kind: 'verification_target',
    domain: 'audio',
    task: 'audio_verification',
    backend: 'verifier_service',
    display_name: '',
    description: '',
    provider: 'aws',
    family: 'Cloud Audio',
    model_ref: 'AWS Transcribe',
    settings: { service_name: 'AWS Transcribe', provider: 'aws' },
    compatibility: ['verification_target', 'benchmark_target'],
    aliases: [],
    item_ids: [],
    credential_profile_id: '',
    enabled: true,
    sort_order: 0,
  },
}

export default function ModelsPage() {
  const [catalog, setCatalog] = useState(null)
  const [providers, setProviders] = useState({})
  const [tab, setTab] = useState('image')
  const [search, setSearch] = useState('')
  const [busyId, setBusyId] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [editorState, setEditorState] = useState(null)
  const [testStatus, setTestStatus] = useState({})

  const reload = useCallback(async () => {
    try {
      setError('')
      const [nextCatalog, nextProviders] = await Promise.all([getModelCatalog(), getConfig()])
      setCatalog(nextCatalog)
      setProviders(nextProviders || {})
    } catch (err) {
      setError(err.message || 'Failed to load models')
    }
  }, [])

  useEffect(() => {
    reload()
  }, [reload])

  const imageSections = useMemo(() => ({
    models: filterEntries(catalog?.source_models?.image || [], search),
    targets: filterEntries(catalog?.verification_targets?.image || [], search),
  }), [catalog, search])

  const audioSections = useMemo(() => ({
    classifiers: filterEntries(catalog?.source_models?.audio_classification || [], search),
    asr: filterEntries(catalog?.source_models?.asr || [], search),
    targets: filterEntries(catalog?.verification_targets?.audio || [], search),
  }), [catalog, search])

  const totals = useMemo(() => {
    return {
      image: (catalog?.source_models?.image || []).length + (catalog?.verification_targets?.image || []).length,
      audio:
        (catalog?.source_models?.audio_classification || []).length +
        (catalog?.source_models?.asr || []).length +
        (catalog?.verification_targets?.audio || []).length,
    }
  }, [catalog])

  const credentialProfiles = useMemo(() => {
    const map = {}
    Object.entries(providers || {}).forEach(([providerId, providerData]) => {
      map[providerId] = (providerData.items || []).map(item => ({
        id: item.id,
        label: item.label || item.id,
        active: providerData.active_id === item.id,
      }))
    })
    return map
  }, [providers])

  const openCreate = useCallback((templateKey) => {
    setMessage('')
    setError('')
    setEditorState({
      mode: 'create',
      templateKey,
      draft: structuredClone(DEFAULTS[templateKey]),
    })
  }, [])

  const openEdit = useCallback((entry) => {
    setMessage('')
    setError('')
    setEditorState({
      mode: 'edit',
      templateKey: inferTemplate(entry),
      entryId: entry.id,
      draft: entryToDraft(entry),
    })
  }, [])

  const handleSave = useCallback(async (draft) => {
    try {
      setBusyId(editorState?.entryId || 'create')
      setError('')
      setMessage('')
      if (editorState?.mode === 'edit' && editorState.entryId) {
        await updateModelEntry(editorState.entryId, normalizeDraftPayload(draft))
        setMessage('Model entry updated.')
      } else {
        await createModelEntry(normalizeDraftPayload(draft))
        setMessage('Model entry created.')
      }
      setEditorState(null)
      await reload()
    } catch (err) {
      setError(err.message || 'Failed to save model entry')
    } finally {
      setBusyId('')
    }
  }, [editorState, reload])

  const handleToggle = useCallback(async (entry) => {
    try {
      setBusyId(entry.id)
      setMessage('')
      setError('')
      await toggleModelEntry(entry.id, !entry.enabled)
      setMessage(`${entry.display_name} ${entry.enabled ? 'disabled' : 'enabled'}.`)
      await reload()
    } catch (err) {
      setError(err.message || 'Failed to toggle model entry')
    } finally {
      setBusyId('')
    }
  }, [reload])

  const handleDuplicate = useCallback(async (entry) => {
    try {
      setBusyId(entry.id)
      setMessage('')
      setError('')
      await duplicateModelEntry(entry.id)
      setMessage(`Duplicated ${entry.display_name}.`)
      await reload()
    } catch (err) {
      setError(err.message || 'Failed to duplicate model entry')
    } finally {
      setBusyId('')
    }
  }, [reload])

  const handleDelete = useCallback(async (entry) => {
    if (!window.confirm(`Delete "${entry.display_name}"? Referenced items will be archived instead of removed.`)) return
    try {
      setBusyId(entry.id)
      setMessage('')
      setError('')
      const response = await deleteModelEntry(entry.id)
      if (response.action === 'archived') {
        setMessage(`${entry.display_name} was archived because existing jobs or history still reference it.`)
      } else {
        setMessage(`${entry.display_name} was deleted.`)
      }
      await reload()
    } catch (err) {
      setError(err.message || 'Failed to delete model entry')
    } finally {
      setBusyId('')
    }
  }, [reload])

  const handleTest = useCallback(async (entry) => {
    try {
      setBusyId(entry.id)
      setTestStatus(prev => ({ ...prev, [entry.id]: { state: 'running', message: 'Testing…' } }))
      const result = await testModelEntry(entry.id)
      setTestStatus(prev => ({
        ...prev,
        [entry.id]: {
          state: result.ok ? 'ok' : 'error',
          message: result.message || (result.ok ? 'Validation passed' : 'Validation failed'),
        },
      }))
    } catch (err) {
      setTestStatus(prev => ({
        ...prev,
        [entry.id]: {
          state: 'error',
          message: err.message || 'Validation failed',
        },
      }))
    } finally {
      setBusyId('')
    }
  }, [])

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Models</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Manage image and audio entries in one place. Compatibility flags decide where each local model appears across the app.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <SummaryChip label="Image" value={totals.image} />
          <SummaryChip label="Audio" value={totals.audio} />
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {TAB_OPTIONS.map(option => (
          <button
            key={option.id}
            onClick={() => setTab(option.id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              tab === option.id
                ? 'bg-[var(--accent)]/15 border border-[var(--accent)]/30 text-[var(--accent)]'
                : 'bg-slate-800/40 border border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            {option.label}
          </button>
        ))}
        <input
          value={search}
          onChange={event => setSearch(event.target.value)}
          placeholder="Search models, providers, or refs…"
          className="ml-auto min-w-[240px] bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 outline-none focus:border-[var(--accent)]"
        />
      </div>

      {message && <p className="text-sm text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2">{message}</p>}
      {error && <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>}

      {tab === 'image' && (
        <div className="space-y-4">
          <SectionBlock title="Image Models" description="Local image models used by classify, attacks, robustness, benchmarks, and local verification when the compatibility flags are enabled." actionLabel="Add Image Model" onAction={() => openCreate('image_source')}>
            <EntryGrid
              entries={imageSections.models}
              busyId={busyId}
              testStatus={testStatus}
              onEdit={openEdit}
              onToggle={handleToggle}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
              onTest={handleTest}
            />
          </SectionBlock>

          <SectionBlock title="Remote Image Verification Targets" description="Remote image verifiers, including Hugging Face API targets and cloud providers.">
            <div className="flex gap-2 mb-3">
              <button onClick={() => openCreate('hf_target')} className="px-3 py-1.5 rounded-lg text-[11px] bg-slate-800/40 border border-slate-700 text-slate-300 hover:text-[var(--accent)] transition">Hugging Face API</button>
              <button onClick={() => openCreate('image_service_target')} className="px-3 py-1.5 rounded-lg text-[11px] bg-slate-800/40 border border-slate-700 text-slate-300 hover:text-[var(--accent)] transition">Cloud Service</button>
            </div>
            <EntryGrid
              entries={imageSections.targets}
              busyId={busyId}
              testStatus={testStatus}
              onEdit={openEdit}
              onToggle={handleToggle}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
              onTest={handleTest}
            />
          </SectionBlock>

        </div>
      )}

      {tab === 'audio' && (
        <div className="space-y-4">
          <SectionBlock title="Audio Classification Models" description="Local classifiers used by audio attack flows." actionLabel="Add Audio Classifier" onAction={() => openCreate('audio_source')}>
            <EntryGrid
              entries={audioSections.classifiers}
              busyId={busyId}
              testStatus={testStatus}
              onEdit={openEdit}
              onToggle={handleToggle}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
              onTest={handleTest}
            />
          </SectionBlock>

          <SectionBlock title="ASR Models" description="Speech-to-text models used by transcription, hidden command, robustness, and benchmark flows." actionLabel="Add ASR Model" onAction={() => openCreate('asr_source')}>
            <EntryGrid
              entries={audioSections.asr}
              busyId={busyId}
              testStatus={testStatus}
              onEdit={openEdit}
              onToggle={handleToggle}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
              onTest={handleTest}
            />
          </SectionBlock>

          <SectionBlock title="Audio Verification Targets" description="Remote ASR verification targets used by audio transfer checks." actionLabel="Add Audio Target" onAction={() => openCreate('audio_service_target')}>
            <EntryGrid
              entries={audioSections.targets}
              busyId={busyId}
              testStatus={testStatus}
              onEdit={openEdit}
              onToggle={handleToggle}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
              onTest={handleTest}
            />
          </SectionBlock>
        </div>
      )}

      {editorState && (
        <ModelEditorModal
          draft={editorState.draft}
          busy={busyId === (editorState.entryId || 'create')}
          mode={editorState.mode}
          providerProfiles={credentialProfiles}
          onClose={() => setEditorState(null)}
          onChange={draft => setEditorState(prev => ({ ...prev, draft }))}
          onSave={() => handleSave(editorState.draft)}
        />
      )}
    </div>
  )
}

function SectionBlock({ title, description, actionLabel, onAction, children }) {
  return (
    <Card>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <SectionLabel>{title}</SectionLabel>
          <p className="text-xs text-slate-500 -mt-1">{description}</p>
        </div>
        {actionLabel && (
          <button
            onClick={onAction}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition"
          >
            {actionLabel}
          </button>
        )}
      </div>
      {children}
    </Card>
  )
}

function EntryGrid({ entries, busyId, testStatus, onEdit, onToggle, onDuplicate, onDelete, onTest }) {
  if (!entries.length) {
    return <p className="text-xs text-slate-500 py-6 text-center">No entries match the current filter.</p>
  }
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
      {entries.map(entry => (
        <EntryCard
          key={entry.id}
          entry={entry}
          busy={busyId === entry.id}
          testStatus={testStatus[entry.id]}
          onEdit={() => onEdit(entry)}
          onToggle={() => onToggle(entry)}
          onDuplicate={() => onDuplicate(entry)}
          onDelete={() => onDelete(entry)}
          onTest={() => onTest(entry)}
        />
      ))}
    </div>
  )
}

function EntryCard({ entry, busy, testStatus, onEdit, onToggle, onDuplicate, onDelete, onTest }) {
  const archived = !!entry.archived_at
  return (
    <div className={`rounded-xl border p-4 transition ${archived ? 'border-amber-500/30 bg-amber-500/5' : 'border-slate-700/60 bg-slate-800/30 hover:border-slate-600'}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="text-sm font-medium text-slate-200 truncate">{entry.display_name}</h4>
            {entry.builtin && <Tag tone="cyan">built-in</Tag>}
            {!entry.enabled && <Tag tone="slate">disabled</Tag>}
            {archived && <Tag tone="amber">archived</Tag>}
          </div>
          <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
            {entry.description || `${entry.family || 'Model'} · ${entry.backend} · ${entry.domain}`}
          </p>
        </div>
        <button
          onClick={onToggle}
          disabled={busy}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${entry.enabled ? 'bg-emerald-500' : 'bg-slate-600'} ${busy ? 'opacity-50' : ''}`}
          title={entry.enabled ? 'Disable entry' : 'Enable entry'}
        >
          <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${entry.enabled ? 'translate-x-[18px]' : 'translate-x-[3px]'}`} />
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        <MetaPill>{entry.kind.replace(/_/g, ' ')}</MetaPill>
        <MetaPill>{entry.task.replace(/_/g, ' ')}</MetaPill>
        <MetaPill>{entry.backend}</MetaPill>
        {entry.provider && <MetaPill>{entry.provider}</MetaPill>}
      </div>

      {entry.model_ref && (
        <div className="mt-3 text-[11px] text-slate-400 font-mono truncate">
          {entry.model_ref}
        </div>
      )}

      {entry.compatibility?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {entry.compatibility.map(flag => <Tag key={flag} tone="slate">{compatibilityLabel(flag)}</Tag>)}
        </div>
      )}

      {entry.item_ids?.length > 0 && (
        <p className="mt-3 text-[11px] text-slate-500">
          {entry.item_ids.length} linked item{entry.item_ids.length !== 1 ? 's' : ''}
        </p>
      )}

      {testStatus && (
        <div className={`mt-3 px-2.5 py-2 rounded-lg text-[11px] border ${
          testStatus.state === 'ok'
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
            : testStatus.state === 'running'
              ? 'bg-slate-700/40 border-slate-600/30 text-slate-300'
              : 'bg-red-500/10 border-red-500/20 text-red-300'
        }`}>
          {testStatus.message}
        </div>
      )}

      <div className="mt-4 flex items-center gap-2 flex-wrap">
        <button onClick={onTest} disabled={busy} className={ACTION_BUTTON_CLASS}>Test</button>
        <button onClick={onDuplicate} disabled={busy} className={ACTION_BUTTON_CLASS}>Duplicate</button>
        <button onClick={onEdit} disabled={busy} className={ACTION_BUTTON_CLASS}>Edit</button>
        <button onClick={onDelete} disabled={busy} className={ACTION_BUTTON_DANGER_CLASS}>Delete</button>
      </div>
    </div>
  )
}

function ModelEditorModal({ draft, busy, mode, providerProfiles, onClose, onChange, onSave }) {
  const providerOptions = Object.keys(providerProfiles || {})
  const profileOptions = providerProfiles[draft.provider] || []
  const serviceOptions = draft.domain === 'audio' ? SERVICE_OPTIONS.audio : SERVICE_OPTIONS.image
  const trustRemoteCode = !!draft.settings?.trust_remote_code

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-2xl border border-slate-700 bg-slate-900 p-6">
        <div className="flex items-start justify-between gap-3 mb-5">
          <div>
            <h4 className="text-base font-semibold text-slate-100">{mode === 'edit' ? 'Edit Model Entry' : 'Create Model Entry'}</h4>
            <p className="text-xs text-slate-500 mt-1">Stable registry IDs are generated automatically. Queued jobs keep their own model snapshot after submission.</p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition text-xl leading-none">&times;</button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Field label="Display name">
            <input value={draft.display_name} onChange={event => onChange({ ...draft, display_name: event.target.value })}
              className={INPUT_CLASS} placeholder="e.g. Custom ViT Large" />
          </Field>

          <Field label="Description">
            <input value={draft.description || ''} onChange={event => onChange({ ...draft, description: event.target.value })}
              className={INPUT_CLASS} placeholder="Optional note for other researchers" />
          </Field>

          <Field label="Kind">
            <select value={draft.kind} onChange={event => onChange(reseedForKind(draft, event.target.value))} className={INPUT_CLASS}>
              <option value="source_model">Source model</option>
              <option value="verification_target">Verification target</option>
            </select>
          </Field>

          <Field label="Domain">
            <select value={draft.domain} onChange={event => onChange(adjustDomain(draft, event.target.value))} className={INPUT_CLASS}>
              <option value="image">Image</option>
              <option value="audio">Audio</option>
            </select>
          </Field>

          <Field label="Task">
            <select value={draft.task} onChange={event => onChange(adjustTask(draft, event.target.value))} className={INPUT_CLASS}>
              {draft.kind === 'source_model' && draft.domain === 'image' && <option value="image_classification">Image classification</option>}
              {draft.kind === 'source_model' && draft.domain === 'audio' && <option value="audio_classification">Audio classification</option>}
              {draft.kind === 'source_model' && draft.domain === 'audio' && <option value="asr">ASR</option>}
              {draft.kind === 'verification_target' && draft.domain === 'image' && <option value="image_verification">Image verification</option>}
              {draft.kind === 'verification_target' && draft.domain === 'audio' && <option value="audio_verification">Audio verification</option>}
            </select>
          </Field>

          <Field label="Backend">
            <select value={draft.backend} onChange={event => onChange(adjustBackend(draft, event.target.value))} className={INPUT_CLASS}>
              {draft.kind === 'source_model' && draft.domain === 'image' && <option value="hf_local_image">Hugging Face local image</option>}
              {draft.kind === 'source_model' && draft.task === 'audio_classification' && <option value="hf_local_audio">Hugging Face local audio</option>}
              {draft.kind === 'source_model' && draft.task === 'asr' && <option value="hf_local_asr">Hugging Face local ASR</option>}
              {draft.kind === 'verification_target' && draft.domain === 'image' && <option value="hf_api">Hugging Face API</option>}
              {draft.kind === 'verification_target' && <option value="verifier_service">Cloud service</option>}
            </select>
          </Field>

          <Field label="Provider">
            <input value={draft.provider || ''} onChange={event => onChange({ ...draft, provider: event.target.value })}
              className={INPUT_CLASS} placeholder="e.g. aws, huggingface, openai" list="provider-options" />
            <datalist id="provider-options">
              {providerOptions.map(option => <option key={option} value={option} />)}
            </datalist>
          </Field>

          <Field label="Family">
            <input value={draft.family || ''} onChange={event => onChange({ ...draft, family: event.target.value })}
              className={INPUT_CLASS} placeholder="e.g. CNN, Whisper, Cloud Image" />
          </Field>

          {draft.backend !== 'verifier_service' && (
            <Field label="Model ref">
              <input value={draft.model_ref || ''} onChange={event => onChange({ ...draft, model_ref: event.target.value })}
                className={INPUT_CLASS} placeholder="Hugging Face repo or model ID" />
            </Field>
          )}

          {draft.kind === 'verification_target' && draft.backend === 'verifier_service' && (
            <Field label="Service name">
              <select
                value={draft.settings?.service_name || draft.model_ref || ''}
                onChange={event => onChange({
                  ...draft,
                  model_ref: event.target.value,
                  display_name: draft.display_name || event.target.value,
                  settings: { ...(draft.settings || {}), service_name: event.target.value, provider: draft.provider },
                })}
                className={INPUT_CLASS}
              >
                {serviceOptions.map(option => <option key={option} value={option}>{option}</option>)}
              </select>
            </Field>
          )}

          <Field label="Credential profile">
            <select
              value={draft.credential_profile_id || ''}
              onChange={event => onChange({ ...draft, credential_profile_id: event.target.value })}
              className={INPUT_CLASS}
            >
              <option value="">No linked profile</option>
              {profileOptions.map(option => (
                <option key={option.id} value={option.id}>
                  {option.label}{option.active ? ' (active)' : ''}
                </option>
              ))}
            </select>
            {draft.provider && PROVIDER_HINTS[draft.provider] && (
              <p className="text-[11px] text-slate-500 mt-1">{PROVIDER_HINTS[draft.provider]}</p>
            )}
          </Field>

          {draft.kind === 'source_model' && (
            <label className="flex items-center gap-2 text-xs text-slate-300 mt-6">
              <input
                type="checkbox"
                checked={trustRemoteCode}
                onChange={event => onChange({
                  ...draft,
                  settings: { ...(draft.settings || {}), trust_remote_code: event.target.checked },
                })}
                className="accent-[var(--accent)]"
              />
              Allow `trust_remote_code`
            </label>
          )}
        </div>

        {draft.kind === 'source_model' && (
          <div className="mt-5">
            <SectionLabel>Compatibility</SectionLabel>
            <div className="flex flex-wrap gap-2">
              {compatibilityChoices(draft).map(choice => (
                <label key={choice.value} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-700 text-xs text-slate-300 bg-slate-800/30">
                  <input
                    type="checkbox"
                    checked={(draft.compatibility || []).includes(choice.value)}
                    onChange={() => onChange(toggleCompatibility(draft, choice.value))}
                    className="accent-[var(--accent)]"
                  />
                  {choice.label}
                </label>
              ))}
            </div>
          </div>
        )}

        <div className="mt-6 flex items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={!!draft.enabled}
              onChange={event => onChange({ ...draft, enabled: event.target.checked })}
              className="accent-[var(--accent)]"
            />
            Enabled
          </label>
          <div className="flex items-center gap-2">
            <button onClick={onClose} className="px-4 py-2 rounded-lg border border-slate-700 text-slate-400 hover:text-slate-200 transition">Cancel</button>
            <button onClick={onSave} disabled={busy || !draft.display_name.trim()} className="px-4 py-2 rounded-lg bg-[var(--accent)] text-slate-900 font-semibold hover:brightness-110 disabled:opacity-50 transition">
              {busy ? 'Saving…' : mode === 'edit' ? 'Save Changes' : 'Create Entry'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function filterEntries(entries, search) {
  const query = search.trim().toLowerCase()
  if (!query) return entries
  return entries.filter(entry => {
    const haystack = [
      entry.display_name,
      entry.description,
      entry.provider,
      entry.family,
      entry.model_ref,
      ...(entry.compatibility || []),
    ].join(' ').toLowerCase()
    return haystack.includes(query)
  })
}

function entryToDraft(entry) {
  return {
    ...entry,
    settings: { ...(entry.settings || {}) },
    compatibility: [...(entry.compatibility || [])],
    aliases: [...(entry.aliases || [])],
    item_ids: [...(entry.item_ids || [])],
    credential_profile_id: entry.credential_profile_id || '',
  }
}

function inferTemplate(entry) {
  if (entry.kind === 'source_model' && entry.domain === 'image') return 'image_source'
  if (entry.kind === 'source_model' && entry.task === 'audio_classification') return 'audio_source'
  if (entry.kind === 'source_model' && entry.task === 'asr') return 'asr_source'
  if (entry.kind === 'verification_target' && entry.backend === 'hf_api') return 'hf_target'
  if (entry.kind === 'verification_target' && entry.domain === 'image') return 'image_service_target'
  return 'audio_service_target'
}

function normalizeDraftPayload(draft) {
  const payload = {
    kind: draft.kind,
    domain: draft.domain,
    task: draft.task,
    backend: draft.backend,
    display_name: draft.display_name.trim(),
    description: (draft.description || '').trim(),
    provider: (draft.provider || '').trim(),
    family: (draft.family || '').trim(),
    model_ref: (draft.model_ref || '').trim(),
    settings: { ...(draft.settings || {}) },
    compatibility: [...(draft.compatibility || [])],
    aliases: [...(draft.aliases || [])],
    item_ids: [...(draft.item_ids || [])],
    credential_profile_id: draft.credential_profile_id || null,
    enabled: !!draft.enabled,
    sort_order: Number(draft.sort_order || 0),
  }
  if (draft.backend === 'verifier_service') {
    payload.settings.service_name = draft.settings?.service_name || draft.model_ref || draft.display_name
    payload.settings.provider = draft.provider || draft.settings?.provider || ''
    payload.model_ref = payload.settings.service_name
  }
  return payload
}

function compatibilityChoices(draft) {
  if (draft.task === 'image_classification') {
    return [
      { value: 'classify', label: 'Usable for classify' },
      { value: 'attack_source', label: 'Usable for attack source' },
      { value: 'robustness_source', label: 'Usable for robustness' },
      { value: 'benchmark_source', label: 'Usable for benchmark source' },
      { value: 'verification_target_local', label: 'Usable for local verification' },
      { value: 'benchmark_target_local', label: 'Usable for benchmark target' },
    ]
  }
  if (draft.task === 'audio_classification') {
    return [{ value: 'attack_source', label: 'Usable for audio attack source' }]
  }
  if (draft.task === 'asr') {
    return [
      { value: 'attack_source', label: 'Usable for ASR attack source' },
      { value: 'robustness_source', label: 'Usable for robustness' },
      { value: 'benchmark_source', label: 'Usable for benchmark source' },
    ]
  }
  return []
}

function toggleCompatibility(draft, value) {
  const current = new Set(draft.compatibility || [])
  if (current.has(value)) current.delete(value)
  else current.add(value)
  return { ...draft, compatibility: Array.from(current) }
}

function reseedForKind(draft, nextKind) {
  if (nextKind === draft.kind) return draft
  if (nextKind === 'source_model') return { ...structuredClone(DEFAULTS.image_source), display_name: draft.display_name, description: draft.description }
  return { ...structuredClone(DEFAULTS.hf_target), display_name: draft.display_name, description: draft.description }
}

function adjustDomain(draft, nextDomain) {
  if (draft.kind === 'source_model') {
    if (nextDomain === 'image') return { ...structuredClone(DEFAULTS.image_source), display_name: draft.display_name, description: draft.description }
    return { ...structuredClone(DEFAULTS.audio_source), display_name: draft.display_name, description: draft.description }
  }
  if (nextDomain === 'audio') return { ...structuredClone(DEFAULTS.audio_service_target), display_name: draft.display_name, description: draft.description }
  return { ...structuredClone(DEFAULTS.hf_target), display_name: draft.display_name, description: draft.description }
}

function adjustTask(draft, nextTask) {
  if (draft.kind !== 'source_model') return { ...draft, task: nextTask }
  if (nextTask === 'image_classification') return { ...structuredClone(DEFAULTS.image_source), display_name: draft.display_name, description: draft.description }
  if (nextTask === 'asr') return { ...structuredClone(DEFAULTS.asr_source), display_name: draft.display_name, description: draft.description }
  return { ...structuredClone(DEFAULTS.audio_source), display_name: draft.display_name, description: draft.description }
}

function adjustBackend(draft, nextBackend) {
  if (nextBackend === 'hf_api') return { ...structuredClone(DEFAULTS.hf_target), display_name: draft.display_name, description: draft.description }
  if (nextBackend === 'verifier_service' && draft.domain === 'audio') return { ...structuredClone(DEFAULTS.audio_service_target), display_name: draft.display_name, description: draft.description }
  if (nextBackend === 'verifier_service') return { ...structuredClone(DEFAULTS.image_service_target), display_name: draft.display_name, description: draft.description }
  if (nextBackend === 'hf_local_asr') return { ...structuredClone(DEFAULTS.asr_source), display_name: draft.display_name, description: draft.description }
  if (nextBackend === 'hf_local_audio') return { ...structuredClone(DEFAULTS.audio_source), display_name: draft.display_name, description: draft.description }
  return { ...structuredClone(DEFAULTS.image_source), display_name: draft.display_name, description: draft.description }
}

function compatibilityLabel(flag) {
  return flag.replace(/_/g, ' ')
}

function SummaryChip({ label, value }) {
  return (
    <span className="text-[10px] px-2 py-1 rounded-md bg-slate-800 border border-slate-700/50 text-slate-400">
      {label}: {value}
    </span>
  )
}

function Tag({ tone = 'slate', children }) {
  const styles = {
    cyan: 'bg-cyan-500/10 border-cyan-500/20 text-cyan-300',
    amber: 'bg-amber-500/10 border-amber-500/20 text-amber-300',
    slate: 'bg-slate-800/60 border-slate-700 text-slate-400',
  }
  return <span className={`text-[10px] px-2 py-0.5 rounded-full border ${styles[tone] || styles.slate}`}>{children}</span>
}

function MetaPill({ children }) {
  return <span className="text-[10px] px-2 py-1 rounded-md bg-slate-900/70 border border-slate-700/60 text-slate-500">{children}</span>
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="block text-xs text-slate-400 mb-1.5">{label}</span>
      {children}
    </label>
  )
}
