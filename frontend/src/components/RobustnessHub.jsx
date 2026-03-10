import { useNavigate, useLocation } from 'react-router-dom'
import RobustnessPage from './RobustnessPage'
import AttackBenchmarkPage from './AttackBenchmarkPage'

const TABS = [
  { id: 'model', label: 'Model Robustness' },
  { id: 'attack', label: 'Attack Robustness' },
]

export default function RobustnessHub() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const active = pathname.endsWith('/attack') ? 'attack' : 'model'

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-slate-200">Robustness Analysis</h2>
        <p className="text-xs text-slate-500 mt-0.5">Evaluate model resilience and attack effectiveness</p>
      </div>

      <div className="flex items-stretch gap-0 bg-slate-800/40 border border-slate-700/50 rounded-lg p-0.5 max-w-md">
        {TABS.map(t => {
          const isActive = t.id === active
          return (
            <button
              key={t.id}
              onClick={() => navigate(`/robustness/${t.id}`, { replace: true })}
              className={`
                flex-1 px-4 py-2 rounded-md text-xs font-semibold transition-all duration-150
                ${isActive
                  ? 'bg-[var(--accent)]/15 text-[var(--accent)] shadow-sm'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700/30'
                }
              `}
            >
              {t.label}
            </button>
          )
        })}
      </div>

      {active === 'model' ? <RobustnessPage /> : <AttackBenchmarkPage />}
    </div>
  )
}
