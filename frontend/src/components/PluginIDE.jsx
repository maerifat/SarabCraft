import { useState, useRef, useCallback, useEffect } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { vscodeDark } from '@uiw/codemirror-theme-vscode'
import { runPluginPlayground, savePluginCode, updatePluginCode, getPluginCode, generateTTS } from '../api/client'

const DEFAULT_CODE = `PLUGIN_NAME = "My Plugin"
PLUGIN_TYPE = "image"
PLUGIN_DESCRIPTION = ""

def classify(adversarial_image, *, original_image=None, config={}):
    """
    adversarial_image : PIL.Image.Image
    original_image    : PIL.Image.Image or None
    config            : dict — global variables from Settings > Variables
    Must return: [{"label": str, "confidence": float}, ...]
    """
    return [{"label": "example", "confidence": 0.99}]
`

export default function PluginIDE({ pluginId = null, onClose, onSaved }) {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [filename, setFilename] = useState('my_plugin')
  const [loading, setLoading] = useState(!!pluginId)
  const [tab, setTab] = useState('editor')

  const [inputMode, setInputMode] = useState('upload')
  const [imagePreview, setImagePreview] = useState(null)
  const [imageB64, setImageB64] = useState(null)
  const [imageUrl, setImageUrl] = useState('')
  const [audioB64, setAudioB64] = useState(null)
  const [audioPreview, setAudioPreview] = useState(null)
  const [ttsText, setTtsText] = useState('')
  const [ttsLoading, setTtsLoading] = useState(false)

  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState('')

  const fileRef = useRef()
  const audioFileRef = useRef()
  const isEdit = !!pluginId

  useEffect(() => {
    if (!pluginId) return
    getPluginCode(pluginId)
      .then(d => { setCode(d.code); setFilename(d.file?.replace('.py', '') || ''); setLoading(false) })
      .catch(() => setLoading(false))
  }, [pluginId])

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose?.()
        return
      }
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key === 's') {
        e.preventDefault()
        handleSave()
        return
      }
      if (mod && e.key === 'Enter') {
        e.preventDefault()
        handleRun()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose, code, filename])

  const handleImageUpload = useCallback(e => {
    const file = e.target?.files?.[0] || e.dataTransfer?.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result.split(',')[1]
      setImageB64(base64)
      setImagePreview(reader.result)
      setAudioB64(null)
      setAudioPreview(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleAudioUpload = useCallback(e => {
    const file = e.target?.files?.[0] || e.dataTransfer?.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      setAudioB64(reader.result.split(',')[1])
      setAudioPreview(prev => {
        if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
        return URL.createObjectURL(file)
      })
      setImageB64(null)
      setImagePreview(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleTTS = async () => {
    if (!ttsText.trim()) return
    setTtsLoading(true)
    try {
      const res = await generateTTS(ttsText)
      if (res.audio_b64) {
        setAudioB64(res.audio_b64)
        setAudioPreview(`data:audio/wav;base64,${res.audio_b64}`)
        setImageB64(null)
        setImagePreview(null)
      }
    } catch (e) { setResult({ error: e.message, predictions: [] }) }
    finally { setTtsLoading(false) }
  }

  const handleRun = async () => {
    setRunning(true)
    setResult(null)
    const body = { code }
    if (inputMode === 'upload' && imageB64) body.image_b64 = imageB64
    else if (inputMode === 'url' && imageUrl.trim()) body.image_url = imageUrl.trim()
    else if ((inputMode === 'audio' || inputMode === 'tts') && audioB64) body.audio_b64 = audioB64
    try { setResult(await runPluginPlayground(body)) }
    catch (e) {
      try { setResult(JSON.parse(e.message)) }
      catch { setResult({ error: e.message, predictions: [] }) }
    }
    finally { setRunning(false) }
  }

  const handleSave = async () => {
    setSaving(true)
    setSaveMsg('')
    try {
      if (isEdit) {
        await updatePluginCode(pluginId, code)
        setSaveMsg('Saved!')
      } else {
        await savePluginCode(filename, code)
        setSaveMsg('Plugin created!')
      }
      onSaved?.()
    } catch (e) { setSaveMsg(e.message) }
    finally { setSaving(false) }
  }

  if (loading) return (
    <div className="fixed inset-0 z-50 bg-slate-950/90 flex items-center justify-center">
      <span className="text-sm text-slate-400 animate-pulse">Loading plugin...</span>
    </div>
  )

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/95 flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>
            <span className="text-xs font-semibold text-slate-200">Plugin IDE</span>
          </div>
          <div className="h-4 w-px bg-slate-700" />
          {!isEdit && (
            <div className="flex items-center gap-1.5">
              <input value={filename} onChange={e => setFilename(e.target.value)}
                className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-300 font-mono outline-none focus:border-[var(--accent)] w-36" placeholder="filename" />
              <span className="text-xs text-slate-600 font-mono">.py</span>
            </div>
          )}
          {isEdit && <span className="text-xs text-slate-500 font-mono">{filename}.py</span>}

          {/* Tabs */}
          <div className="flex bg-slate-800 rounded-lg p-0.5 ml-4">
            {['editor', 'playground'].map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                  tab === t ? 'bg-slate-700 text-[var(--accent)]' : 'text-slate-500 hover:text-slate-300'
                }`}>
                {t === 'editor' ? 'Editor' : 'Playground'}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {saveMsg && (
            <span className={`text-xs ${saveMsg.includes('!') ? 'text-green-400' : 'text-red-400'}`}>
              {saveMsg}
            </span>
          )}
          <button onClick={handleSave} disabled={saving}
            className="px-4 py-1.5 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-bold rounded-lg disabled:opacity-40 text-xs transition">
            {saving ? 'Saving...' : isEdit ? 'Save Changes' : 'Save Plugin'}
          </button>
          <button onClick={onClose} className="p-1.5 text-slate-500 hover:text-slate-300 transition rounded-lg hover:bg-slate-800">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Editor panel — always visible but can shrink */}
        <div className={`flex flex-col border-r border-slate-800 transition-all ${tab === 'playground' ? 'w-1/2' : 'w-full'}`}>
          <div className="flex items-center px-3 py-1.5 bg-slate-900/50 border-b border-slate-800/50">
            <span className="text-[10px] text-slate-500 font-mono">{filename || 'untitled'}.py</span>
            <span className="ml-auto text-[9px] text-slate-600">Python</span>
          </div>
          <div className="flex-1 overflow-auto">
            <CodeMirror
              value={code}
              onChange={setCode}
              theme={vscodeDark}
              extensions={[python()]}
              basicSetup={{
                lineNumbers: true,
                foldGutter: true,
                bracketMatching: true,
                closeBrackets: true,
                autocompletion: true,
                highlightActiveLine: true,
                indentOnInput: true,
              }}
              style={{ height: '100%', fontSize: '13px' }}
            />
          </div>
        </div>

        {/* Playground panel */}
        {tab === 'playground' && (
          <div className="w-1/2 flex flex-col bg-slate-950 overflow-hidden">
            {/* Input section */}
            <div className="border-b border-slate-800 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300">Input</span>
                <div className="flex bg-slate-800 rounded-lg p-0.5">
                  {[
                    { id: 'upload', label: 'Image' },
                    { id: 'url', label: 'Image URL' },
                    { id: 'audio', label: 'Audio' },
                    { id: 'tts', label: 'TTS' },
                  ].map(m => (
                    <button key={m.id} onClick={() => setInputMode(m.id)}
                      className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition ${
                        inputMode === m.id ? 'bg-slate-700 text-[var(--accent)]' : 'text-slate-500 hover:text-slate-300'
                      }`}>
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>

              {inputMode === 'upload' && (
                <div className="space-y-2">
                  <div
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => { e.preventDefault(); handleImageUpload(e) }}
                    onClick={() => fileRef.current?.click()}
                    className="border border-dashed border-slate-700 rounded-lg p-4 text-center cursor-pointer hover:border-[var(--accent)] transition">
                    {imagePreview ? (
                      <img src={imagePreview} alt="preview" className="max-h-32 mx-auto rounded" />
                    ) : (
                      <div>
                        <svg className="w-6 h-6 text-slate-600 mx-auto mb-1" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.41a2.25 2.25 0 013.182 0l2.909 2.91m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" /></svg>
                        <span className="text-[10px] text-slate-500">Drop image or click to browse</span>
                      </div>
                    )}
                  </div>
                  <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                </div>
              )}

              {inputMode === 'url' && (
                <input value={imageUrl} onChange={e => setImageUrl(e.target.value)}
                  placeholder="https://example.com/image.jpg"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-300 font-mono placeholder-slate-600 outline-none focus:border-[var(--accent)]" />
              )}

              {inputMode === 'audio' && (
                <div className="space-y-2">
                  <div
                    onClick={() => audioFileRef.current?.click()}
                    className="border border-dashed border-slate-700 rounded-lg p-4 text-center cursor-pointer hover:border-[var(--accent)] transition">
                    {audioPreview ? (
                      <audio src={audioPreview} controls className="mx-auto" style={{ height: 32 }} />
                    ) : (
                      <div>
                        <svg className="w-6 h-6 text-slate-600 mx-auto mb-1" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" /></svg>
                        <span className="text-[10px] text-slate-500">Click to upload audio file</span>
                      </div>
                    )}
                  </div>
                  <input ref={audioFileRef} type="file" accept="audio/*" className="hidden" onChange={handleAudioUpload} />
                </div>
              )}

              {inputMode === 'tts' && (
                <div className="space-y-2">
                  <textarea value={ttsText} onChange={e => setTtsText(e.target.value)} rows={2}
                    placeholder="Type text to convert to speech..."
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-300 placeholder-slate-600 outline-none focus:border-[var(--accent)] resize-none" />
                  <button onClick={handleTTS} disabled={ttsLoading || !ttsText.trim()}
                    className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white font-medium rounded-lg text-xs disabled:opacity-40 transition">
                    {ttsLoading ? 'Generating...' : 'Generate Audio'}
                  </button>
                  {audioPreview && <audio src={audioPreview} controls className="w-full" style={{ height: 32 }} />}
                </div>
              )}

              <button onClick={handleRun} disabled={running}
                className="w-full flex items-center justify-center gap-2 py-2.5 bg-green-600 hover:bg-green-500 text-white font-bold rounded-lg text-xs disabled:opacity-40 transition">
                {running ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" /></svg>
                    Run
                  </>
                )}
              </button>
            </div>

            {/* Output section */}
            <div className="flex-1 overflow-auto p-4">
              {!result && !running && (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center">
                    <svg className="w-10 h-10 text-slate-700 mx-auto mb-2" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" /></svg>
                    <p className="text-xs text-slate-600">Click Run to test your plugin</p>
                    <p className="text-[10px] text-slate-700 mt-1">Upload input or run with a default gray image</p>
                  </div>
                </div>
              )}

              {result && <PlaygroundResult result={result} />}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}


function PlaygroundResult({ result }) {
  const hasError = !!result.error
  const hasPreds = result.predictions?.length > 0

  return (
    <div className="space-y-4">
      {/* Status header */}
      <div className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${
        hasError ? 'bg-red-500/5 border-red-500/20' : 'bg-green-500/5 border-green-500/20'
      }`}>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
          hasError ? 'bg-red-500/15' : 'bg-green-500/15'
        }`}>
          {hasError ? (
            <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" /></svg>
          ) : (
            <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          )}
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-slate-200">
            {hasError ? 'Error' : 'Success'}
            {result.plugin_name && <span className="text-slate-500 font-normal ml-2">{result.plugin_name}</span>}
          </div>
          {result.elapsed_ms != null && (
            <span className="text-[10px] text-slate-500">{Math.round(result.elapsed_ms)}ms</span>
          )}
        </div>
        {hasPreds && (
          <span className="text-[10px] px-2 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20">
            {result.predictions.length} prediction{result.predictions.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Error message */}
      {hasError && (
        <div className="space-y-2">
          <div className="bg-red-500/5 border border-red-500/15 rounded-lg p-3">
            <p className="text-xs text-red-400 font-mono whitespace-pre-wrap">{result.error}</p>
          </div>
          {result.traceback && (
            <details className="group">
              <summary className="text-[10px] text-slate-500 cursor-pointer hover:text-slate-400 transition">
                Show traceback
              </summary>
              <pre className="mt-2 bg-slate-900 border border-slate-800 rounded-lg p-3 text-[10px] text-red-400/80 font-mono overflow-x-auto whitespace-pre leading-relaxed">
                {result.traceback}
              </pre>
            </details>
          )}
        </div>
      )}

      {/* Predictions */}
      {hasPreds && (
        <div className="space-y-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">Predictions</span>
          <div className="space-y-1">
            {result.predictions.map((p, i) => (
              <div key={i} className="flex items-center gap-3 px-3 py-2 bg-slate-900/60 rounded-lg border border-slate-800/50">
                <span className="text-[10px] text-slate-600 w-4 text-right font-mono">{i + 1}</span>
                <span className="flex-1 text-xs text-slate-200 font-medium truncate">{p.label}</span>
                <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-[var(--accent)] rounded-full transition-all"
                    style={{ width: `${Math.min(100, (p.confidence || 0) * 100)}%` }} />
                </div>
                <span className="text-[11px] text-slate-400 font-mono w-14 text-right">
                  {((p.confidence || 0) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Raw JSON */}
      <details className="group">
        <summary className="text-[10px] text-slate-500 cursor-pointer hover:text-slate-400 transition flex items-center gap-1.5">
          <svg className="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
          Raw Response
        </summary>
        <pre className="mt-2 bg-slate-900 border border-slate-800 rounded-lg p-4 text-[11px] text-slate-400 font-mono overflow-x-auto whitespace-pre leading-relaxed">
          {JSON.stringify(result, null, 2)}
        </pre>
      </details>
    </div>
  )
}
