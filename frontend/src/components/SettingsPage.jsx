import { useParams, useNavigate } from 'react-router-dom'
import Config from './Config'
import ModelsPage from './ModelsPage'
import PluginsManager from './PluginsManager'
import VariablesPage from './VariablesPage'
import DocsPage from './DocsPage'

const SECTIONS = [
  {
    id: 'credentials',
    label: 'Credentials',
    description: 'Service profiles',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z" />
      </svg>
    ),
  },
  {
    id: 'variables',
    label: 'Variables',
    description: 'Global key-value store',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.745 3A23.933 23.933 0 003 12c0 3.183.62 6.22 1.745 9M19.255 3C20.38 5.78 21 8.817 21 12s-.62 6.22-1.745 9m-13.46-1.5c-.58-2.296-.92-4.79-.92-7.5s.34-5.204.92-7.5m13.46 0c.58 2.296.92 4.79.92 7.5s-.34 5.204-.92 7.5M12 3v18" />
      </svg>
    ),
  },
  {
    id: 'models',
    label: 'Models',
    description: 'Source and target registry',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5m-16.5 5.25h16.5m-16.5 5.25h16.5M5.25 4.5v15m13.5-15v15" />
      </svg>
    ),
  },
  {
    id: 'plugins',
    label: 'Plugins',
    description: 'Custom classifiers',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 16.875h3.375m0 0h3.375m-3.375 0V13.5m0 3.375v3.375M6 10.5h2.25a2.25 2.25 0 002.25-2.25V6a2.25 2.25 0 00-2.25-2.25H6A2.25 2.25 0 003.75 6v2.25A2.25 2.25 0 006 10.5zm0 9.75h2.25A2.25 2.25 0 0010.5 18v-2.25a2.25 2.25 0 00-2.25-2.25H6a2.25 2.25 0 00-2.25 2.25V18A2.25 2.25 0 006 20.25zm9.75-9.75H18a2.25 2.25 0 002.25-2.25V6A2.25 2.25 0 0018 3.75h-2.25A2.25 2.25 0 0013.5 6v2.25a2.25 2.25 0 002.25 2.25z" />
      </svg>
    ),
  },
  {
    id: 'docs',
    label: 'Documentation',
    description: 'Plugin developer guide',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
      </svg>
    ),
  },
]

export default function SettingsPage() {
  const { tab } = useParams()
  const navigate = useNavigate()
  const active = SECTIONS.find(s => s.id === tab) ? tab : 'credentials'

  return (
    <div className="flex gap-6 min-h-[480px]">
      {/* Sidebar */}
      <nav className="w-48 flex-shrink-0">
        <div className="sticky top-4 space-y-1">
          {SECTIONS.map(s => (
            <button
              key={s.id}
              onClick={() => navigate(`/settings/${s.id}`)}
              className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-left transition ${
                active === s.id
                  ? 'bg-slate-700/70 text-[var(--accent)] border border-[var(--accent)]/30'
                  : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-transparent'
              }`}
            >
              <span className={active === s.id ? 'text-[var(--accent)]' : 'text-slate-500'}>{s.icon}</span>
              <div>
                <div className="text-xs font-medium leading-tight">{s.label}</div>
                <div className="text-[10px] text-slate-500 leading-tight mt-0.5">{s.description}</div>
              </div>
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {active === 'credentials' && <Config />}
        {active === 'variables' && <VariablesPage />}
        {active === 'models' && <ModelsPage />}
        {active === 'plugins' && <PluginsManager />}
        {active === 'docs' && <DocsPage />}
      </div>
    </div>
  )
}
