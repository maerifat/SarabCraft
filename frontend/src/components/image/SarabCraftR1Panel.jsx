import Slider from '../ui/Slider'
import { Select } from '../ui/Section'
import { SARABCRAFT_R1_STRATEGY_OPTIONS, sarabCraftR1StrategyHint } from './sarabcraftR1'

const BASE_PARAM_KEYS = [
  'epsilon',
  'iterations',
  'momentum_decay',
  'kernel_size',
  'cfm_mix_prob',
  'cfm_mix_upper',
]

export default function SarabCraftR1Panel({ params, getParam, setParam }) {
  const strategy = getParam('r1_multi_image_strategy') || 'tile_shuffle'
  const multiImage = Boolean(getParam('r1_multi_image'))
  const strategyMeta = params.r1_multi_image_strategy || {}
  const countMeta = params.r1_multi_image_count || {}

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {BASE_PARAM_KEYS.map((key) => {
          const meta = params[key]
          if (!meta) return null
          return (
            <Slider
              key={key}
              label={meta.label}
              value={getParam(key)}
              onChange={(value) => setParam(key, value)}
              min={meta.min}
              max={meta.max}
              step={meta.step}
              defaultValue={meta.default}
            />
          )
        })}
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/30 p-4 space-y-3">
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={multiImage}
            onChange={(event) => setParam('r1_multi_image', event.target.checked)}
            className="mt-0.5 w-4 h-4 rounded accent-[var(--accent)]"
          />
          <div>
            <div className="text-sm font-medium text-slate-200">Use multi-image transfer mode</div>
            <p className="text-xs text-slate-500 mt-1">
              Unlock broader bank-building strategies for harder transfer experiments. Leave it off for the standard R1 path.
            </p>
          </div>
        </label>

        {multiImage && (
          <div className="space-y-3 pt-3 border-t border-slate-700/50">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-end">
              <Select
                label={strategyMeta.label || 'Multi-image strategy'}
                value={strategy}
                onChange={(value) => setParam('r1_multi_image_strategy', value)}
                options={SARABCRAFT_R1_STRATEGY_OPTIONS}
                className="min-w-0"
              />
              <Slider
                label={countMeta.label || 'Image count'}
                value={getParam('r1_multi_image_count') || countMeta.default || 10}
                onChange={(value) => setParam('r1_multi_image_count', value)}
                min={countMeta.min ?? 2}
                max={countMeta.max ?? 30}
                step={countMeta.step ?? 1}
                defaultValue={countMeta.default ?? 10}
              />
            </div>
            <p className="text-[11px] text-slate-500">
              {sarabCraftR1StrategyHint(strategy)}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
