import { useState, useEffect, useCallback, useRef } from 'react'
import { getPlugins, deletePlugin, testPlugin, uploadPlugin, togglePlugin, changePluginType } from '../api/client'
import { ErrorMsg } from './ui/Section'
import PluginIDE from './PluginIDE'

const TYPE_STYLE = {
  image: { bg: 'bg-blue-500/15', text: 'text-blue-400', border: 'border-blue-500/20' },
  audio: { bg: 'bg-purple-500/15', text: 'text-purple-400', border: 'border-purple-500/20' },
  both:  { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/20' },
}

export default function PluginsManager() {
  const [plugins, setPlugins] = useState([])
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [addOpen, setAddOpen] = useState(false)
  const [ideOpen, setIdeOpen] = useState(null)
  const addRef = useRef()

  const reload = useCallback(async () => {
    try { setPlugins((await getPlugins()).plugins || []) }
    catch (e) { setError(e.message) }
  }, [])

  useEffect(() => { reload() }, [reload])
  useEffect(() => {
    const close = e => { if (addRef.current && !addRef.current.contains(e.target)) setAddOpen(false) }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const filtered = filter
    ? plugins.filter(p => p.name.toLowerCase().includes(filter.toLowerCase()) || (p.description||'').toLowerCase().includes(filter.toLowerCase()))
    : plugins

  return (
    <div className="space-y-5">
      <ErrorMsg msg={error} />

      {/* Header */}
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Plugins</h3>
          <p className="text-xs text-slate-500 mt-0.5">Extend transfer verification with custom Python classifiers.</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative" ref={addRef}>
            <button onClick={() => setAddOpen(v => !v)} className="flex items-center gap-1.5 text-xs px-4 py-2 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-bold rounded-lg transition">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
              Add Plugin
            </button>
            {addOpen && (
              <div className="absolute right-0 mt-1 w-52 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-20 overflow-hidden">
                <button onClick={() => { setIdeOpen({ mode: 'new' }); setAddOpen(false) }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-slate-700/60 transition">
                  <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>
                  <div>
                    <div className="text-xs font-medium text-slate-200">Open IDE</div>
                    <div className="text-[10px] text-slate-500">Write & test with playground</div>
                  </div>
                </button>
                <button onClick={() => { setIdeOpen({ mode: 'upload' }); setAddOpen(false) }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-slate-700/60 transition">
                  <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>
                  <div>
                    <div className="text-xs font-medium text-slate-200">Upload .py / .zip</div>
                    <div className="text-[10px] text-slate-500">Upload plugin files</div>
                  </div>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats + filter */}
      {plugins.length > 0 && (
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-[10px] px-2 py-1 rounded-md bg-slate-800 text-slate-400 border border-slate-700/50">
            {plugins.filter(p => p.enabled).length}/{plugins.length} enabled
          </span>
          {plugins.length > 3 && (
            <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Search plugins..."
              className="ml-auto bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 outline-none focus:border-[var(--accent)] w-48 transition" />
          )}
        </div>
      )}

      {/* Upload form */}
      {ideOpen?.mode === 'upload' && <UploadForm onDone={() => { setIdeOpen(null); reload() }} onCancel={() => setIdeOpen(null)} />}

      {/* Plugin Grid */}
      {filtered.length === 0 && !ideOpen ? (
        <EmptyState onAdd={() => setAddOpen(true)} hasPlugins={plugins.length > 0} />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {filtered.map(p => (
            <PluginCard key={p.id} plugin={p} onRefresh={reload}
              onOpenIDE={() => setIdeOpen({ mode: 'edit', pluginId: p.id })} />
          ))}
        </div>
      )}

      {/* IDE overlay */}
      {ideOpen?.mode === 'new' && (
        <PluginIDE onClose={() => setIdeOpen(null)} onSaved={() => { setIdeOpen(null); reload() }} />
      )}
      {ideOpen?.mode === 'edit' && (
        <PluginIDE pluginId={ideOpen.pluginId} onClose={() => setIdeOpen(null)} onSaved={reload} />
      )}
    </div>
  )
}

function EmptyState({ onAdd, hasPlugins }) {
  if (hasPlugins) return <p className="text-xs text-slate-500 text-center py-6">No plugins match your search.</p>
  return (
    <div className="text-center py-12 px-6">
      <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center mx-auto mb-4">
        <svg className="w-7 h-7 text-slate-600" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 002.25-2.25V6a2.25 2.25 0 00-2.25-2.25H6A2.25 2.25 0 003.75 6v2.25A2.25 2.25 0 006 10.5zm0 9.75h2.25A2.25 2.25 0 0010.5 18v-2.25a2.25 2.25 0 00-2.25-2.25H6a2.25 2.25 0 00-2.25 2.25V18A2.25 2.25 0 006 20.25zm9.75-9.75H18a2.25 2.25 0 002.25-2.25V6A2.25 2.25 0 0018 3.75h-2.25A2.25 2.25 0 0013.5 6v2.25a2.25 2.25 0 002.25 2.25z" /></svg>
      </div>
      <h4 className="text-sm font-medium text-slate-300 mb-1">No plugins installed</h4>
      <p className="text-xs text-slate-500 mb-5 max-w-xs mx-auto">Add custom classifiers to extend transfer verification with your own models or external APIs.</p>
      <button onClick={onAdd} className="px-5 py-2.5 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-bold rounded-lg text-sm transition">
        Add Your First Plugin
      </button>
    </div>
  )
}

function PluginCard({ plugin, onRefresh, onOpenIDE }) {
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [toggling, setToggling] = useState(false)
  const [typeMenuOpen, setTypeMenuOpen] = useState(false)
  const [cardError, setCardError] = useState('')
  const menuRef = useRef()
  const typeRef = useRef()

  useEffect(() => {
    const close = e => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false)
      if (typeRef.current && !typeRef.current.contains(e.target)) setTypeMenuOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const handleTest = async () => {
    setTesting(true); setTestResult(null)
    try { setTestResult(await testPlugin(plugin.id)) }
    catch (e) { setTestResult({ ok: false, error: e.message }) }
    finally { setTesting(false) }
  }
  const handleDelete = async () => {
    if (!confirm(`Delete "${plugin.name}"?`)) return
    try { await deletePlugin(plugin.id); onRefresh() }
    catch (e) { setCardError(e.message) }
  }
  const handleToggle = async () => {
    setToggling(true)
    try { await togglePlugin(plugin.id, !plugin.enabled); onRefresh() }
    catch (e) { setCardError(e.message) }
    finally { setToggling(false) }
  }
  const handleTypeChange = async (newType) => {
    setTypeMenuOpen(false)
    try { await changePluginType(plugin.id, newType); onRefresh() }
    catch (e) { setCardError(e.message) }
  }

  const ts = TYPE_STYLE[plugin.type] || { bg: 'bg-slate-700/30', text: 'text-slate-400', border: 'border-slate-600' }
  const hasError = !!plugin.error

  return (
    <div className={`relative rounded-xl border transition-all ${hasError ? 'border-red-500/30 bg-red-500/5' : !plugin.enabled ? 'border-slate-700/40 bg-slate-800/15 opacity-60' : 'border-slate-700/60 bg-slate-800/30 hover:border-slate-600'}`}>
      <div className="p-4">
        <div className="flex items-start gap-3">
          <div className={`w-10 h-10 rounded-lg ${ts.bg} border ${ts.border} flex items-center justify-center flex-shrink-0`}>
            <svg className={`w-5 h-5 ${ts.text}`} fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-slate-200 truncate">{plugin.name}</span>
              <div className="relative" ref={typeRef}>
                <button onClick={() => setTypeMenuOpen(v => !v)}
                  className={`text-[9px] px-1.5 py-0.5 rounded font-semibold uppercase tracking-wide cursor-pointer hover:ring-1 hover:ring-slate-500 transition ${ts.bg} ${ts.text}`}
                  title="Click to change type">
                  {plugin.type}
                  <svg className="inline w-2.5 h-2.5 ml-0.5 -mt-px" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
                </button>
                {typeMenuOpen && (
                  <div className="absolute left-0 mt-1 w-28 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-30 overflow-hidden">
                    {['image', 'audio', 'both'].map(t => {
                      const s = TYPE_STYLE[t] || {}
                      return (
                        <button key={t} onClick={() => handleTypeChange(t)}
                          className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-slate-700/60 transition ${plugin.type === t ? 'bg-slate-700/40' : ''}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${s.bg || 'bg-slate-600'}`} style={{ background: t === 'image' ? '#3b82f6' : t === 'audio' ? '#a855f7' : '#10b981' }} />
                          <span className={`uppercase font-semibold tracking-wide ${s.text || 'text-slate-400'}`}>{t}</span>
                          {plugin.type === t && <svg className="w-3 h-3 ml-auto text-green-400" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
            {plugin.description && <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-2">{plugin.description}</p>}
          </div>
          <div className="relative" ref={menuRef}>
            <button onClick={() => setMenuOpen(v => !v)} className="p-1 text-slate-500 hover:text-slate-300 transition rounded">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M10 6a2 2 0 110-4 2 2 0 010 4zm0 6a2 2 0 110-4 2 2 0 010 4zm0 6a2 2 0 110-4 2 2 0 010 4z" /></svg>
            </button>
            {menuOpen && (
              <div className="absolute right-0 mt-1 w-44 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-20 overflow-hidden">
                <button onClick={() => { handleTest(); setMenuOpen(false) }} className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-300 hover:bg-slate-700/60 transition">
                  <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" /></svg>
                  Quick Test
                </button>
                <button onClick={() => { onOpenIDE(); setMenuOpen(false) }} className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-300 hover:bg-slate-700/60 transition">
                  <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>
                  Open in IDE
                </button>
                <div className="border-t border-slate-700/50" />
                <button onClick={() => { handleDelete(); setMenuOpen(false) }} className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" /></svg>
                  Delete
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="mt-3 flex items-center gap-2 flex-wrap text-[10px]">
          {plugin.file && <span className="text-slate-600 font-mono">{plugin.file}</span>}
          {hasError && (
            <span className="inline-flex items-center gap-1 text-red-400">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" /></svg>
              {plugin.error}
            </span>
          )}
        </div>

        <div className="mt-2.5 flex items-center gap-2">
          <button onClick={handleToggle} disabled={toggling || hasError}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${plugin.enabled && !hasError ? 'bg-green-500' : 'bg-slate-600'} ${toggling ? 'opacity-50' : ''}`}
            title={plugin.enabled ? 'Enabled — click to disable' : 'Disabled — click to enable'}>
            <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${plugin.enabled && !hasError ? 'translate-x-[18px]' : 'translate-x-[3px]'}`} />
          </button>
          <span className={`text-[10px] ${hasError ? 'text-red-400' : plugin.enabled ? 'text-green-400' : 'text-slate-500'}`}>
            {hasError ? 'Error' : plugin.enabled ? 'Enabled' : 'Disabled'}
          </span>
          {testing && <span className="text-[10px] text-slate-500 animate-pulse">Testing...</span>}
          {testResult && (
            <span className={`text-[10px] px-2 py-0.5 rounded ${testResult.ok ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
              {testResult.ok ? `Passed · ${Math.round(testResult.elapsed_ms)}ms` : testResult.error || 'Failed'}
            </span>
          )}
          <button onClick={onOpenIDE}
            className="ml-auto text-[10px] text-slate-500 hover:text-[var(--accent)] transition flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>
            Open IDE
          </button>
        </div>

        {cardError && (
          <div className="mt-2 flex items-center justify-between gap-2 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded-lg">
            <span className="text-[11px] text-red-400 truncate">{cardError}</span>
            <button onClick={() => setCardError('')} className="text-red-400 hover:text-red-300 flex-shrink-0 transition">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function UploadForm({ onDone, onCancel }) {
  const ref = useRef()
  const [uploading, setUploading] = useState(false)
  const [msg, setMsg] = useState('')
  const [dragOver, setDragOver] = useState(false)

  const upload = async (file) => {
    if (!file) return
    if (!file.name.endsWith('.py') && !file.name.endsWith('.zip')) { setMsg('Only .py and .zip files'); return }
    setUploading(true); setMsg('')
    try {
      const fd = new FormData(); fd.append('file', file)
      await uploadPlugin(fd)
      onDone()
    } catch (e) { setMsg(e.message) }
    finally { setUploading(false) }
  }

  const onDrop = e => { e.preventDefault(); setDragOver(false); upload(e.dataTransfer.files[0]) }

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-800/30 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/50">
        <span className="text-xs font-semibold text-slate-200">Upload Plugin</span>
        <button onClick={onCancel} className="text-slate-500 hover:text-slate-300 transition">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>
      <div className="p-4">
        <div onDragOver={e => { e.preventDefault(); setDragOver(true) }} onDragLeave={() => setDragOver(false)} onDrop={onDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition ${dragOver ? 'border-[var(--accent)] bg-[var(--accent)]/5' : 'border-slate-700'}`}>
          <svg className="w-8 h-8 text-slate-600 mx-auto mb-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>
          <p className="text-sm text-slate-400 mb-1">Drag & drop a <code className="bg-slate-700/60 px-1.5 py-0.5 rounded text-xs">.py</code> or <code className="bg-slate-700/60 px-1.5 py-0.5 rounded text-xs">.zip</code> file</p>
          <p className="text-[10px] text-slate-600 mb-4">or browse to select a file</p>
          <input ref={ref} type="file" accept=".py,.zip" className="hidden" onChange={e => upload(e.target.files[0])} />
          <button onClick={() => ref.current?.click()} disabled={uploading} className="px-5 py-2 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-bold rounded-lg disabled:opacity-40 text-sm transition">
            {uploading ? 'Uploading...' : 'Browse Files'}
          </button>
        </div>
        <p className="text-[10px] text-slate-600 mt-3">Files must export <code className="text-slate-400">PLUGIN_NAME</code> and <code className="text-slate-400">classify()</code>.</p>
        {msg && <p className="text-xs text-red-400 mt-2">{msg}</p>}
      </div>
    </div>
  )
}
