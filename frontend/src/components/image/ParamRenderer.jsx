import Slider from '../ui/Slider'
import SarabCraftR1Panel from './SarabCraftR1Panel'
import { SARABCRAFT_R1_NAME } from './sarabcraftR1'

export default function ParamRenderer({ attackName, params, getParam, setParam }) {
  if (attackName === SARABCRAFT_R1_NAME) {
    return <SarabCraftR1Panel params={params} getParam={getParam} setParam={setParam} />
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      {Object.entries(params).map(([key, meta]) => {
        if (meta.type === 'bool') {
          return (
            <label key={key} className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
              <input type="checkbox" checked={!!getParam(key)} onChange={e => setParam(key, e.target.checked)} className="accent-[var(--accent)]" />
              {meta.label}
            </label>
          )
        }
        if (meta.type === 'select') {
          return (
            <div key={key}>
              <label className="text-[10px] text-slate-500 mb-0.5 block">{meta.label}</label>
              <select value={getParam(key)} onChange={e => setParam(key, e.target.value)}
                className="w-full px-2 py-1.5 bg-slate-800 border border-slate-700 rounded text-xs text-slate-200">
                {(meta.options || []).map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
          )
        }
        return (
          <Slider key={key} label={meta.label} value={getParam(key)} onChange={v => setParam(key, v)}
            min={meta.min} max={meta.max} step={meta.step} defaultValue={meta.default} />
        )
      })}
    </div>
  )
}
