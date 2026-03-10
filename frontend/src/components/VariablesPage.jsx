import { useState, useEffect, useCallback } from 'react'
import { getVariables, setVariable, deleteVariable } from '../api/client'
import { Card, ErrorMsg } from './ui/Section'

export default function VariablesPage() {
  const [vars, setVars] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')

  const [newKey, setNewKey] = useState('')
  const [newVal, setNewVal] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newMasked, setNewMasked] = useState(false)
  const [adding, setAdding] = useState(false)

  const load = useCallback(async () => {
    try {
      const d = await getVariables()
      setVars(d.variables || [])
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const handleAdd = async () => {
    if (!newKey.trim()) return
    setAdding(true); setError('')
    try {
      await setVariable(newKey.trim(), newVal, newMasked, newDesc)
      setNewKey(''); setNewVal(''); setNewDesc(''); setNewMasked(false)
      load()
    } catch (e) { setError(e.message) }
    finally { setAdding(false) }
  }

  const handleDelete = async (key) => {
    if (!confirm(`Delete variable "${key}"?`)) return
    setError('')
    try { await deleteVariable(key); load() }
    catch (e) { setError(e.message) }
  }

  const filtered = filter
    ? vars.filter(v => v.key.toLowerCase().includes(filter.toLowerCase()) || (v.description || '').toLowerCase().includes(filter.toLowerCase()))
    : vars

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-slate-200">Global Variables</h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Centralized key-value store shared across all plugins. Access in plugin code via <code className="bg-slate-800 px-1 rounded text-slate-400">config["KEY"]</code>.
        </p>
      </div>

      <ErrorMsg msg={error} />

      {/* Add Variable Form */}
      <Card>
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-4 h-4 text-[var(--accent)]" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
          <span className="text-xs font-semibold text-slate-200">Add Variable</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="text-[10px] text-slate-500 block mb-1">Key <span className="text-red-400">*</span></label>
            <input
              value={newKey} onChange={e => setNewKey(e.target.value)}
              placeholder="API_KEY"
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono placeholder-slate-600 outline-none focus:border-[var(--accent)] transition"
              onKeyDown={e => e.key === 'Enter' && handleAdd()}
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-500 block mb-1">Value <span className="text-red-400">*</span></label>
            <input
              value={newVal} onChange={e => setNewVal(e.target.value)}
              placeholder="sk-..."
              type={newMasked ? 'password' : 'text'}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono placeholder-slate-600 outline-none focus:border-[var(--accent)] transition"
              onKeyDown={e => e.key === 'Enter' && handleAdd()}
            />
          </div>
          <div className="md:col-span-2">
            <label className="text-[10px] text-slate-500 block mb-1">Description <span className="text-slate-600">(optional)</span></label>
            <input
              value={newDesc} onChange={e => setNewDesc(e.target.value)}
              placeholder="API key for external classifier service"
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-[var(--accent)] transition"
            />
          </div>
        </div>
        <div className="flex items-center justify-between mt-3">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input type="checkbox" checked={newMasked} onChange={e => setNewMasked(e.target.checked)} className="w-3.5 h-3.5 accent-[var(--accent)]" />
            <span className="text-xs text-slate-400">Mask value</span>
            <span className="text-[10px] text-slate-600">(hidden in UI, still available to plugins)</span>
          </label>
          <button
            onClick={handleAdd} disabled={adding || !newKey.trim()}
            className="px-5 py-2 bg-[var(--accent)] hover:bg-cyan-500 text-slate-900 font-bold rounded-lg disabled:opacity-40 text-sm transition"
          >
            {adding ? 'Adding...' : 'Add Variable'}
          </button>
        </div>
      </Card>

      {/* Variable List */}
      <Card>
        <div className="flex items-center justify-between mb-3 gap-3">
          <span className="text-xs font-semibold text-slate-200 flex items-center gap-2">
            Variables
            <span className="text-[10px] font-normal text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">{vars.length}</span>
          </span>
          {vars.length > 3 && (
            <input
              value={filter} onChange={e => setFilter(e.target.value)}
              placeholder="Filter..."
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 outline-none focus:border-[var(--accent)] w-48 transition"
            />
          )}
        </div>

        {loading ? (
          <p className="text-xs text-slate-500 py-4 text-center">Loading...</p>
        ) : filtered.length === 0 ? (
          <div className="text-center py-8">
            <svg className="w-8 h-8 text-slate-700 mx-auto mb-2" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M4.745 3A23.933 23.933 0 003 12c0 3.183.62 6.22 1.745 9M19.255 3C20.38 5.78 21 8.817 21 12s-.62 6.22-1.745 9m-13.46-1.5c-.58-2.296-.92-4.79-.92-7.5s.34-5.204.92-7.5m13.46 0c.58 2.296.92 4.79.92 7.5s-.34 5.204-.92 7.5M12 3v18" /></svg>
            <p className="text-xs text-slate-500">{vars.length === 0 ? 'No variables yet.' : 'No matches.'}</p>
            {vars.length === 0 && <p className="text-[10px] text-slate-600 mt-1">Add API keys, tokens, or endpoints above.</p>}
          </div>
        ) : (
          <div className="space-y-1.5">
            {filtered.map(v => (
              <div key={v.key} className="flex items-center gap-3 px-3 py-2.5 bg-slate-800/50 rounded-lg border border-slate-700/50 group hover:border-slate-600 transition">
                <code className="text-xs text-[var(--accent)] font-mono flex-shrink-0 min-w-[100px]">{v.key}</code>
                <span className="text-xs text-slate-600">=</span>
                <span className={`text-xs flex-1 truncate font-mono ${v.masked ? 'text-slate-600 select-none' : 'text-slate-300'}`}>
                  {v.value}
                </span>
                {v.description && (
                  <span className="text-[10px] text-slate-500 truncate max-w-[200px] hidden md:block" title={v.description}>{v.description}</span>
                )}
                {v.masked && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 flex-shrink-0">masked</span>
                )}
                <button
                  onClick={() => handleDelete(v.key)}
                  className="text-slate-600 hover:text-red-400 transition p-1 opacity-0 group-hover:opacity-100"
                  title="Delete variable"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
