import { useState, useEffect, useRef, useCallback } from 'react'
import { generateTTS } from '../../api/client'
import { Card, SectionLabel } from '../ui/Section'
import { downloadB64 } from '../../utils/download'

export const TTS_VOICES = {
  'Jenny (US Female)': 'en-US-JennyNeural', 'Guy (US Male)': 'en-US-GuyNeural',
  'Aria (US Female)': 'en-US-AriaNeural', 'Ryan (UK Male)': 'en-GB-RyanNeural',
  'Sonia (UK Female)': 'en-GB-SoniaNeural', 'Natasha (AU Female)': 'en-AU-NatashaNeural',
}
const voiceOpts = Object.entries(TTS_VOICES).map(([k, v]) => ({ value: v, label: k }))

/* ── helpers ─────────────────────────────────────────────── */

function formatBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(1)} MB`
}

function formatDuration(s) {
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`
  const m = Math.floor(s / 60), sec = (s % 60).toFixed(1)
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}

async function decodeAudioFile(file) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  try {
    const buf = await file.arrayBuffer()
    return await ctx.decodeAudioData(buf)
  } finally { ctx.close() }
}

async function decodeB64Audio(b64) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  try {
    return await ctx.decodeAudioData(bytes.buffer)
  } finally { ctx.close() }
}

function downsamplePeaks(channelData, bars) {
  const blockSize = Math.floor(channelData.length / bars)
  const peaks = new Float32Array(bars)
  for (let i = 0; i < bars; i++) {
    let max = 0
    const start = i * blockSize
    for (let j = 0; j < blockSize; j++) {
      const v = Math.abs(channelData[start + j] || 0)
      if (v > max) max = v
    }
    peaks[i] = max
  }
  return peaks
}

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result.split(',')[1])
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/* ── Waveform ────────────────────────────────────────────── */

export function Waveform({ audioBuffer, height = 48, barCount = 80, color = '#06b6d4', className = '' }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!audioBuffer || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = height
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)
    const data = audioBuffer.getChannelData(0)
    const peaks = downsamplePeaks(data, barCount)
    const maxPeak = Math.max(...peaks, 0.01)
    const barW = w / barCount
    const gap = Math.max(1, barW * 0.2)
    for (let i = 0; i < barCount; i++) {
      const amp = peaks[i] / maxPeak
      const barH = Math.max(2, amp * (h - 4))
      const x = i * barW + gap / 2
      const y = (h - barH) / 2
      ctx.fillStyle = color
      ctx.globalAlpha = 0.3 + amp * 0.7
      ctx.beginPath()
      ctx.roundRect(x, y, barW - gap, barH, 1.5)
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }, [audioBuffer, height, barCount, color])

  return <canvas ref={canvasRef} className={`w-full ${className}`} style={{ height }} />
}

function WaveformFromB64({ b64, height = 36, barCount = 64, color = '#06b6d4' }) {
  const [buf, setBuf] = useState(null)
  useEffect(() => {
    if (!b64) { setBuf(null); return }
    decodeB64Audio(b64).then(setBuf).catch(() => setBuf(null))
  }, [b64])
  if (!buf) return null
  return <Waveform audioBuffer={buf} height={height} barCount={barCount} color={color} />
}

/* ── AudioPlayer (with waveform + play + download) ───────── */

export function AudioPlayer({ wavB64, label, filename = 'adversarial_audio.wav', showWaveform = true }) {
  if (!wavB64) return null
  return (
    <div>
      <p className="text-[11px] text-slate-500 mb-1 font-medium uppercase tracking-wide">{label}</p>
      {showWaveform && (
        <div className="mb-1.5 rounded bg-slate-800/40 px-2 py-1 border border-slate-700/30">
          <WaveformFromB64 b64={wavB64} height={32} barCount={64} />
        </div>
      )}
      <div className="flex items-center gap-2">
        <audio controls src={`data:audio/wav;base64,${wavB64}`} className="flex-1 h-8" />
        <button onClick={() => downloadB64(wavB64, filename, 'audio/wav')}
          className="flex items-center gap-1 px-2 py-1 bg-slate-700/60 hover:bg-slate-600 text-slate-300 rounded border border-slate-600 transition text-[10px] shrink-0"
          title="Download WAV">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3 h-3">
            <path d="M10 3a.75.75 0 01.75.75v7.69l2.22-2.22a.75.75 0 011.06 1.06l-3.5 3.5a.75.75 0 01-1.06 0l-3.5-3.5a.75.75 0 111.06-1.06l2.22 2.22V3.75A.75.75 0 0110 3z" />
            <path d="M3 15.75a.75.75 0 01.75-.75h12.5a.75.75 0 010 1.5H3.75a.75.75 0 01-.75-.75z" />
          </svg>
          WAV
        </button>
      </div>
    </div>
  )
}

/* ── AudioInput (tabbed: Upload | TTS | Record) ──────────── */

const INPUT_MODES = [
  { id: 'upload', label: 'Upload', icon: 'M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5' },
  { id: 'tts', label: 'TTS', icon: 'M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z' },
  { id: 'mic', label: 'Record', icon: 'M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z' },
]

export function AudioInput({ label, file, setFile }) {
  const [mode, setMode] = useState('upload')
  const [dragOver, setDragOver] = useState(false)
  const [fileInfo, setFileInfo] = useState(null)
  const [fileBuf, setFileBuf] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const fileRef = useRef(null)

  useEffect(() => {
    if (!file) { setFileInfo(null); setFileBuf(null); setPreviewUrl(p => { if (p) URL.revokeObjectURL(p); return null }); return }
    setPreviewUrl(p => { if (p) URL.revokeObjectURL(p); return URL.createObjectURL(file) })
    decodeAudioFile(file)
      .then(buf => {
        setFileInfo({ duration: buf.duration, sampleRate: buf.sampleRate, channels: buf.numberOfChannels })
        setFileBuf(buf)
      })
      .catch(() => { setFileInfo(null); setFileBuf(null) })
    return () => setPreviewUrl(p => { if (p) URL.revokeObjectURL(p); return null })
  }, [file])

  const handleDrop = e => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f && f.type.startsWith('audio/')) setFile(f)
  }

  return (
    <div className="flex-1 min-w-[220px]">
      <span className="block text-xs text-slate-400 mb-1.5">{label}</span>

      {/* File loaded state -- compact summary bar */}
      {file ? (
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 overflow-hidden">
          <div className="flex items-center gap-2 px-2.5 py-1.5">
            <svg className="w-3.5 h-3.5 text-green-400 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-xs text-slate-300 truncate flex-1" title={file.name}>{file.name}</span>
            {fileInfo && (
              <span className="text-[10px] text-slate-500 shrink-0">
                {formatDuration(fileInfo.duration)} · {(fileInfo.sampleRate / 1000).toFixed(0)}kHz · {formatBytes(file.size)}
              </span>
            )}
            <button onClick={() => setFile(null)} className="text-slate-500 hover:text-slate-300 transition p-0.5 shrink-0" title="Remove">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          {fileBuf && (
            <div className="px-2">
              <Waveform audioBuffer={fileBuf} height={28} barCount={48} color="#06b6d4" />
            </div>
          )}
          {previewUrl && (
            <div className="px-2 pb-2 pt-1">
              <audio controls src={previewUrl} className="w-full h-8" />
            </div>
          )}
        </div>
      ) : (
        /* No file -- show tabbed input */
        <div className="rounded-lg border border-slate-700/50 bg-slate-800/20 overflow-hidden">
          {/* Mode tabs */}
          <div className="flex border-b border-slate-700/40">
            {INPUT_MODES.map(m => (
              <button key={m.id} onClick={() => setMode(m.id)}
                className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 text-[11px] font-medium transition
                  ${mode === m.id ? 'text-[var(--accent)] bg-slate-800/60 border-b border-[var(--accent)]' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}
                `}>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d={m.icon} />
                </svg>
                {m.label}
              </button>
            ))}
          </div>

          {/* Mode content */}
          <div className="p-2.5">
            {mode === 'upload' && (
              <UploadPane dragOver={dragOver} setDragOver={setDragOver} onDrop={handleDrop} fileRef={fileRef} setFile={setFile} />
            )}
            {mode === 'tts' && <TTSPane setFile={setFile} />}
            {mode === 'mic' && <MicPane setFile={setFile} />}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Upload pane ─────────────────────────────────────────── */

function UploadPane({ dragOver, setDragOver, onDrop, fileRef, setFile }) {
  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => fileRef.current?.click()}
      className={`cursor-pointer rounded border-2 border-dashed transition-all px-3 py-3 text-center
        ${dragOver ? 'border-[var(--accent)] bg-[var(--accent)]/5' : 'border-slate-700/40 hover:border-slate-600'}
      `}
    >
      <input ref={fileRef} type="file" accept="audio/*" onChange={e => setFile(e.target.files?.[0])} className="hidden" />
      <svg className="w-5 h-5 mx-auto mb-1 text-slate-500" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
      <span className="text-[11px] text-slate-500">Drop audio file or click to browse</span>
    </div>
  )
}

/* ── TTS pane ────────────────────────────────────────────── */

function TTSPane({ setFile }) {
  const [text, setText] = useState('')
  const [voice, setVoice] = useState('en-US-JennyNeural')
  const [busy, setBusy] = useState(false)
  const gen = async () => {
    if (!text.trim()) return; setBusy(true)
    try {
      const r = await generateTTS(text, voice)
      if (r?.wav_b64) {
        const b = atob(r.wav_b64), a = new Uint8Array(b.length)
        for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i)
        setFile(new File([new Blob([a], { type: 'audio/wav' })], 'tts.wav', { type: 'audio/wav' }))
      }
    } catch (e) { console.error(e) } finally { setBusy(false) }
  }
  return (
    <div className="space-y-2">
      <input type="text" value={text} onChange={e => setText(e.target.value)} placeholder="Type text to synthesize…"
        onKeyDown={e => { if (e.key === 'Enter') gen() }}
        className="w-full bg-slate-700/80 border border-slate-600 rounded px-2.5 py-1.5 text-sm" />
      <div className="flex gap-2">
        <select value={voice} onChange={e => setVoice(e.target.value)}
          className="flex-1 bg-slate-700/80 border border-slate-600 rounded px-2 py-1 text-[11px]">
          {voiceOpts.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <button onClick={gen} disabled={busy || !text.trim()}
          className="px-3 py-1 bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 text-[var(--accent)] border border-[var(--accent)]/30 rounded text-xs font-medium disabled:opacity-40 transition">
          {busy ? 'Generating…' : 'Generate'}
        </button>
      </div>
    </div>
  )
}

/* ── Mic pane ────────────────────────────────────────────── */

function MicPane({ setFile }) {
  const [state, setState] = useState('idle')
  const [elapsed, setElapsed] = useState(0)
  const [micError, setMicError] = useState('')
  const [previewUrl, setPreviewUrl] = useState(null)
  const mediaRecRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const streamRef = useRef(null)

  const cleanup = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  useEffect(() => cleanup, [cleanup])

  const startRecording = async () => {
    setMicError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const mimeType = MediaRecorder.isTypeSupported?.('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported?.('audio/webm')
          ? 'audio/webm'
          : MediaRecorder.isTypeSupported?.('audio/mp4')
            ? 'audio/mp4'
            : ''
      const mr = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      const actualType = mr.mimeType || 'audio/webm'
      const ext = actualType.includes('mp4') ? 'mp4' : actualType.includes('ogg') ? 'ogg' : 'webm'
      mediaRecRef.current = mr
      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: actualType })
        const file = new File([blob], `recording.${ext}`, { type: actualType })
        setFile(file)
        if (previewUrl) URL.revokeObjectURL(previewUrl)
        setPreviewUrl(URL.createObjectURL(blob))
        stream.getTracks().forEach(t => t.stop())
        setState('done')
      }
      mr.start(100); setState('recording'); setElapsed(0)
      const t0 = Date.now()
      timerRef.current = setInterval(() => setElapsed((Date.now() - t0) / 1000), 100)
    } catch (err) {
      const msg = err?.name === 'NotAllowedError' ? 'Microphone access denied' : `Recording failed: ${err?.message || 'Unknown error'}`
      setMicError(msg)
    }
  }

  const stopRecording = () => { if (timerRef.current) clearInterval(timerRef.current); mediaRecRef.current?.stop() }
  const reset = () => { if (previewUrl) URL.revokeObjectURL(previewUrl); setPreviewUrl(null); setElapsed(0); setState('idle') }

  if (state === 'idle') return (
    <div className="text-center py-2">
      <button onClick={startRecording}
        className="inline-flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg text-sm text-red-300 transition">
        <svg className="w-4 h-4 text-red-400" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="7" /></svg>
        Start Recording
      </button>
      {micError && <p className="text-[11px] text-red-400 mt-1.5">{micError}</p>}
    </div>
  )

  if (state === 'recording') return (
    <div className="flex items-center justify-center gap-3 py-2">
      <span className="w-2.5 h-2.5 rounded-full bg-red-400 recording-pulse" />
      <span className="text-sm text-slate-300 tabular-nums font-medium">{formatDuration(elapsed)}</span>
      <button onClick={stopRecording}
        className="px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded text-xs text-red-300 font-medium transition">
        Stop
      </button>
    </div>
  )

  return (
    <div className="flex items-center gap-2 py-1">
      <audio controls src={previewUrl} className="flex-1 h-8" />
      <button onClick={reset} className="px-2 py-1 bg-slate-700/60 hover:bg-slate-600 border border-slate-600 rounded text-[11px] text-slate-400 transition shrink-0">
        Re-record
      </button>
    </div>
  )
}

/* ── TTSBlock (legacy export for backward compat) ────────── */
export function TTSBlock({ setFile }) { return <TTSPane setFile={setFile} /> }
export function MicRecorder({ setFile }) { return <MicPane setFile={setFile} /> }

/* ── PreviewTranscription ────────────────────────────────── */

export function PreviewTranscription({ file, model }) {
  const [text, setText] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const preview = async () => {
    if (!file || !model) return
    setBusy(true); setErr(''); setText(null)
    try {
      const { transcribeAudio } = await import('../../api/client')
      const fd = new FormData()
      fd.append('audio_file', file); fd.append('model', model)
      const r = await transcribeAudio(fd)
      setText(r.text ?? '(empty)')
    } catch (e) { setErr(e.message || 'Transcription failed') } finally { setBusy(false) }
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <button onClick={preview} disabled={busy || !file || !model}
        className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-700/60 hover:bg-slate-600 border border-slate-600 rounded-lg text-[11px] text-slate-300 transition disabled:opacity-40"
        title="Preview what the model hears before running the full attack">
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" />
        </svg>
        {busy ? 'Transcribing…' : 'Preview'}
      </button>
      {text !== null && (
        <span className="text-xs text-slate-300 bg-slate-800/60 border border-slate-700/40 rounded px-2 py-1 max-w-sm truncate">
          <span className="text-slate-500 mr-1">Hears:</span>"{text}"
        </span>
      )}
      {err && <span className="text-[11px] text-red-400">{err}</span>}
    </div>
  )
}

/* ── Result Components with A/B comparison ───────────────── */

export function AsrResult({ result, onTransfer, originalWavB64 }) {
  return (
    <Card className="border-[var(--accent)]/30"><SectionLabel>Results</SectionLabel>
      <div className="space-y-1.5 mb-4">
        {result.original_text != null && <p className="text-sm text-slate-300">Original: <span className="text-slate-400">{result.original_text}</span></p>}
        <p className="text-sm text-slate-300">Adversarial: <span className="text-green-400 font-medium">{result.adversarial_text}</span></p>
        {(result.target_text || result.command_text) && <p className="text-sm text-slate-300">Target: <span className="text-slate-400">{result.target_text || result.command_text}</span></p>}
        <p className="text-xs text-slate-500">{result.wer !== undefined && `WER: ${(result.wer * 100).toFixed(1)}%`}{result.snr_db !== undefined && ` · SNR: ${result.snr_db.toFixed(1)} dB`}</p>
      </div>
      <AudioComparison originalWavB64={originalWavB64} adversarialWavB64={result.adversarial_wav_b64} />
      {onTransfer && (
        <div className="mt-3">
          <button onClick={onTransfer} className="px-4 py-2 bg-[var(--accent-dim)] border border-[var(--accent)] text-[var(--accent)] font-semibold rounded-lg text-sm hover:bg-cyan-500/20 transition">Test Transfer</button>
        </div>
      )}
    </Card>
  )
}

export function SimpleAsrResult({ result, label = 'Adversarial audio', originalWavB64, onTransfer }) {
  return (
    <Card className="border-[var(--accent)]/30"><SectionLabel>Results</SectionLabel>
      <div className="space-y-1 mb-4">
        {result.original_text != null && (
          <p className="text-sm text-slate-300">Original: <span className="text-slate-400">{result.original_text || '(silence)'}</span></p>
        )}
        {result.adversarial_text !== undefined && (
          <p className="text-sm text-slate-300">ASR output: <span className="text-green-400 font-medium">{result.adversarial_text || '(silence)'}</span></p>
        )}
        {result.snr_db !== undefined && (
          <p className="text-xs text-slate-500">SNR: {result.snr_db.toFixed(1)} dB</p>
        )}
      </div>
      <AudioComparison originalWavB64={originalWavB64} adversarialWavB64={result.adversarial_wav_b64} advLabel={label} />
      {onTransfer && (
        <div className="mt-3">
          <button onClick={onTransfer} className="px-4 py-2 bg-[var(--accent-dim)] border border-[var(--accent)] text-[var(--accent)] font-semibold rounded-lg text-sm hover:bg-cyan-500/20 transition">Test Transfer</button>
        </div>
      )}
    </Card>
  )
}

function AudioComparison({ originalWavB64, adversarialWavB64, advLabel = 'Adversarial audio' }) {
  if (!originalWavB64 && !adversarialWavB64) return null
  if (!originalWavB64) return <AudioPlayer wavB64={adversarialWavB64} label={advLabel} />
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <div className="rounded-lg bg-slate-800/30 border border-slate-700/30 p-2.5">
        <AudioPlayer wavB64={originalWavB64} label="Original" filename="original_audio.wav" />
      </div>
      <div className="rounded-lg bg-slate-800/30 border border-red-500/10 p-2.5">
        <AudioPlayer wavB64={adversarialWavB64} label={advLabel} />
      </div>
    </div>
  )
}
