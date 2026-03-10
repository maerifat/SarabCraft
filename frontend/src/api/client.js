const API_BASE = '/api';
export const API_VERSION = 2;

async function ok(p) {
  const r = await p
  const text = await r.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : {}
  } catch {
    data = text
  }
  if (!r.ok) {
    if (data && typeof data === 'object') throw new Error(data.error || data.detail || `Request failed (${r.status})`)
    throw new Error(data || `Request failed (${r.status})`)
  }
  return data
}
const get = (u, opts = {}) => ok(fetch(`${API_BASE}${u}`, opts));
const post = (u, b) => ok(fetch(`${API_BASE}${u}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }));
const put = (u, b) => ok(fetch(`${API_BASE}${u}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }));
const postFD = (u, fd, opts = {}) => ok(fetch(`${API_BASE}${u}`, { method: 'POST', body: fd, ...opts }));
const del = u => ok(fetch(`${API_BASE}${u}`, { method: 'DELETE' }));

export function createAbortable() {
  const controller = new AbortController()
  return {
    signal: controller.signal,
    abort: () => controller.abort(),
  }
}

function abortError() {
  if (typeof DOMException !== 'undefined') return new DOMException('Aborted', 'AbortError')
  const err = new Error('Aborted')
  err.name = 'AbortError'
  return err
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError())
      return
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener?.('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      signal?.removeEventListener?.('abort', onAbort)
      reject(abortError())
    }
    signal?.addEventListener?.('abort', onAbort, { once: true })
  })
}

async function submitJob(kind, fd, { signal } = {}) {
  return postFD(`/jobs/submit/${kind}`, fd, { signal })
}

async function getJob(jobId, { afterEventId = 0, signal } = {}) {
  return get(`/jobs/${jobId}?after_event_id=${afterEventId}`, { signal })
}

async function cancelJob(jobId) {
  return post(`/jobs/${jobId}/cancel`, {})
}

async function resumeJob(jobId) {
  return post(`/jobs/${jobId}/resume`, {})
}

function attachAbortCancellation(signal, getJobId, { cancelOnAbort = true } = {}) {
  if (!signal || !cancelOnAbort) return () => {}
  const onAbort = () => {
    const jobId = getJobId()
    if (jobId) cancelJob(jobId).catch(() => {})
  }
  signal.addEventListener('abort', onAbort, { once: true })
  return () => signal.removeEventListener('abort', onAbort)
}

async function waitForJob(jobId, { signal } = {}) {
  while (true) {
    if (signal?.aborted) throw abortError()
    const job = await getJob(jobId, { signal })
    if (job.status === 'completed') return job
    if (job.status === 'failed') throw new Error(job.error_message || 'Job failed')
    if (job.status === 'cancelled') throw new Error('Job cancelled')
    await sleep(1000, signal)
  }
}

async function runJob(kind, fd, { signal, onCreated, cancelOnAbort = true } = {}) {
  const created = await submitJob(kind, fd, { signal })
  if (onCreated) onCreated(created)
  let jobId = created.job_id
  const detachAbort = attachAbortCancellation(signal, () => jobId, { cancelOnAbort })
  try {
    const job = await waitForJob(jobId, { signal })
    return job.result
  } finally {
    detachAbort()
  }
}

function streamJob(kind, fd, { onCreated, onInit, onProgress, onResult, onSummary, onError, onDone, signal, cancelOnAbort = true } = {}) {
  const controller = signal ? undefined : new AbortController()
  const sig = signal || controller?.signal
  let stopped = false
  let jobId = null
  let lastEventId = 0

  const emitEvent = (event) => {
    const data = event.payload_json || {}
    if (event.event_type === 'init' && onInit) onInit(data)
    else if (event.event_type === 'progress' && onProgress) onProgress(data)
    else if (event.event_type === 'result' && onResult) onResult(data)
    else if (event.event_type === 'summary' && onSummary) onSummary(data)
    else if (event.event_type === 'error' && onError) onError(new Error(data.message || 'Job failed'))
  }

  const loop = async () => {
    try {
      const created = await submitJob(kind, fd, { signal: sig })
      jobId = created.job_id
      if (onCreated) onCreated(created)
      const detachAbort = attachAbortCancellation(sig, () => jobId, { cancelOnAbort })
      try {
        while (!stopped) {
          if (sig?.aborted) throw abortError()
          const job = await getJob(jobId, { afterEventId: lastEventId, signal: sig })
          for (const event of job.events || []) {
            lastEventId = Math.max(lastEventId, event.id)
            emitEvent(event)
          }
          if (job.status === 'completed') {
            if (onDone) onDone(job.result || {})
            return
          }
          if (job.status === 'failed') {
            throw new Error(job.error_message || 'Job failed')
          }
          if (job.status === 'cancelled') {
            if (!sig?.aborted && onError) onError(new Error('Job cancelled'))
            return
          }
          await sleep(1000, sig)
        }
      } finally {
        detachAbort()
      }
    } catch (err) {
      if (err.name !== 'AbortError' && onError) onError(err)
    }
  }

  loop()

  return {
    abort: () => {
      stopped = true
      if (jobId) cancelJob(jobId).catch(() => {})
      controller?.abort()
    },
  }
}

// Image
export const getModels = () => get('/models/sources?domain=image&task=image_classification');
export const getAttacks = () => get('/attacks/methods');
export const classifyImage = fd => postFD('/attacks/image/classify', fd);
export const runImageAttack = (fd, opts = {}) => runJob('image_attack', fd, opts);

// Verification
export const getVerificationTargets = (domain = 'image') => get(`/models/verification?domain=${encodeURIComponent(domain)}`);
export const getImageVerificationStatus = () => get('/verification/image/status');
export const getImageHeartbeat = () => get('/verification/image/heartbeat');
export const runImageVerification = body => post('/verification/image/run', body);
export const getAudioVerificationStatus = () => get('/verification/audio/status');
export const getAudioHeartbeat = () => get('/verification/audio/heartbeat');
export const runAudioVerification = body => post('/verification/audio/run', body);

// Config — multi-profile
export const getConfig = () => get('/config');
export const addProfile = (provider, label, fields, auth_method = '') => post(`/config/${provider}/add`, { label, fields, auth_method });
export const deleteProfile = (provider, id) => del(`/config/${provider}/${id}`);
export const activateProfile = (provider, id) => post(`/config/${provider}/${id}/activate`, {});
export const testConnection = (provider, id) => post(`/config/${provider}/${id}/test`, {});
export const detectEnvCreds = provider => get(`/config/${provider}/env-detect`);
export const listAwsProfiles = () => get('/config/aws/profiles');
export const listS3Buckets = () => get('/config/aws/buckets');
export const createS3Bucket = () => post('/config/aws/buckets/create', {});
export const selectS3Bucket = bucket => post('/config/aws/buckets/select', { bucket });

// TTS
export const generateTTS = (text, voice = 'en-US-JennyNeural') => post('/tts/generate', { text, voice });

// Audio attacks
export const getAudioModels = () => get('/models/sources?domain=audio&task=audio_classification');
export const getAudioAttacks = () => get('/attacks/audio/methods');
export const getAudioLabels = mk => get(`/attacks/audio/labels/${encodeURIComponent(mk)}`);
export const getAsrModels = () => get('/models/sources?domain=audio&task=asr');
export const runAudioClassificationAttack = fd => runJob('audio_classification', fd);
export const runTranscriptionAttack = fd => runJob('asr_transcription', fd);
export const runHiddenCommandAttack = fd => runJob('asr_hidden_command', fd);
export const runUniversalMutingAttack = fd => runJob('asr_universal_muting', fd);
export const runPsychoacousticAttack = fd => runJob('asr_psychoacoustic', fd);
export const runOverTheAirAttack = fd => runJob('asr_over_the_air', fd);
export const runSpeechJammingAttack = fd => runJob('asr_speech_jamming', fd);
export const transcribeAudio = fd => postFD('/attacks/asr/transcribe', fd);
export const getUA3Models = () => get('/attacks/asr/ua3/models');
export const runUA3Attack = fd => runJob('asr_ua3', fd);

// Plugins (local only)
export const getPlugins = () => get('/plugins/list');
export const getEnabledPlugins = (type = 'image') => get(`/plugins/enabled?type=${type}`);
export const deletePlugin = id => del(`/plugins/${id}`);
export const togglePlugin = (id, enabled) => post(`/plugins/${id}/toggle`, { enabled });
export const changePluginType = (id, type) => post(`/plugins/${id}/type`, { type });
export const testPlugin = id => post(`/plugins/${id}/test`, {});
export const uploadPlugin = fd => postFD('/plugins/upload', fd);
export const savePluginCode = (filename, code) => post('/plugins/code/save', { filename, code });
export const validatePluginCode = code => post('/plugins/code/validate', { code });
export const getPluginCode = id => get(`/plugins/${id}/code`);
export const updatePluginCode = (id, code) => ok(fetch(`${API_BASE}/plugins/${id}/code`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code }) }));
export const runPluginPlayground = body => post('/plugins/playground/run', body);

// Global Variables
export const getVariables = () => get('/variables/list');
export const setVariable = (key, value, masked, description) => post('/variables/set', { key, value, masked, description });
export const deleteVariable = key => del(`/variables/${encodeURIComponent(key)}`);

// Models registry
export const getModelCatalog = () => get('/models/catalog');
export const createModelEntry = body => post('/models', body);
export const updateModelEntry = (id, body) => put(`/models/${encodeURIComponent(id)}`, body);
export const toggleModelEntry = (id, enabled) => post(`/models/${encodeURIComponent(id)}/toggle`, { enabled });
export const duplicateModelEntry = id => post(`/models/${encodeURIComponent(id)}/duplicate`, {});
export const testModelEntry = id => post(`/models/${encodeURIComponent(id)}/test`, {});
export const deleteModelEntry = id => del(`/models/${encodeURIComponent(id)}`);

// History
export const getHistory = (limit = 50, offset = 0, domain = '', attackType = '') => {
  const params = new URLSearchParams({ limit, offset });
  if (domain) params.set('domain', domain);
  if (attackType) params.set('attack_type', attackType);
  return get(`/history/list?${params}`);
};
export const getHistoryEntry = id => get(`/history/${id}`);
export const deleteHistoryEntry = id => del(`/history/${id}`);
export const clearHistory = () => del('/history/');
export const getHistoryStats = () => get('/history/stats/summary');
export const getHeatmapData = () => get('/history/stats/heatmap');

// Jobs
export const getJobs = (limit = 50, status = '') => {
  const params = new URLSearchParams({ limit })
  if (status) params.set('status', status)
  return get(`/jobs?${params}`)
}
export const getJobDetails = (jobId, { afterEventId = 0, signal } = {}) => getJob(jobId, { afterEventId, signal })
export const cancelJobById = jobId => cancelJob(jobId)
export const resumeJobById = jobId => resumeJob(jobId)

// Explainability
export const runGradCAM = body => post('/explainability/gradcam', body);
export const compareGradCAM = body => post('/explainability/gradcam/compare', body);

// System
export const getSystemInfo = () => get('/system/info');

// Batch & Robustness
export const runBatchAttack = (fd, { signal } = {}) => runJob('batch_attack', fd, { signal });
export const runRobustnessComparison = (fd, { signal } = {}) => runJob('image_robustness', fd, { signal });
export const runBatchAttackStream = (fd, callbacks = {}) => streamJob('batch_attack', fd, callbacks)

export function runRobustnessSSE(fd, callbacks = {}) {
  return streamJob('image_robustness', fd, callbacks)
}

export function runAudioRobustnessSSE(fd, callbacks = {}) {
  return streamJob('audio_robustness', fd, callbacks)
}

// Benchmark
export const getBenchmarkAttacks = () => get('/attacks/benchmark/attacks');
export const getBenchmarkPresets = () => get('/attacks/benchmark/presets');

export function runBenchmarkSSE(fd, { onCreated, onInit, onResult, onSummary, onError, onDone, signal } = {}) {
  return streamJob('benchmark', fd, { onCreated, onInit, onResult, onSummary, onError, onDone, signal })
}
