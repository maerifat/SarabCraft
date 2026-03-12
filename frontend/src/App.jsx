import { useState } from 'react'
import { Routes, Route, NavLink, Navigate, useLocation } from 'react-router-dom'
import LandingPage from './components/LandingPage'
import ClassifierTab from './components/ClassifierTab'
import ImageAttackTab from './components/ImageAttackTab'
import TextAttackTab from './components/TextAttackTab'
import AudioTabs from './components/AudioTabs'
import Dashboard from './components/Dashboard'
import HistoryPage from './components/HistoryPage'
import BatchAttackPage from './components/BatchAttackPage'
import JobsPage from './components/JobsPage'
import RobustnessHub from './components/RobustnessHub'
import SettingsPage from './components/SettingsPage'
import NotFoundPage from './components/NotFoundPage'
import { ToastProvider } from './components/ui/Toast'
import { ConfirmProvider } from './components/ui/ConfirmDialog'
import ErrorBoundary from './components/ui/ErrorBoundary'
import { ImageAttackProvider } from './components/ImageAttackContext'

const NAV_GROUPS = [
  {
    label: 'Overview',
    items: [
      { to: '/', label: 'Home', end: true, icon: 'M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25' },
      { to: '/dashboard', label: 'Dashboard', icon: 'M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z' },
    ]
  },
  {
    label: 'Attack Surface',
    items: [
      { to: '/image-attack', label: 'Image Attack', icon: 'M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z' },
      { to: '/text-attack', label: 'Text Attack', icon: 'M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z' },
      { to: '/audio', label: 'Audio Attack', icon: 'M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z' },
      { to: '/batch', label: 'Batch Attack', icon: 'M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z' },
    ]
  },
  {
    label: 'Analysis',
    items: [
      { to: '/classifier', label: 'Classify', icon: 'M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5a1.5 1.5 0 001.5-1.5V5.25a1.5 1.5 0 00-1.5-1.5H3.75a1.5 1.5 0 00-1.5 1.5v14.25c0 .828.672 1.5 1.5 1.5z' },
      { to: '/robustness', label: 'Robustness', icon: 'M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z', matchPrefix: true },
      { to: '/jobs', label: 'Jobs', icon: 'M3.75 5.25h16.5m-16.5 6.75h16.5m-16.5 6.75h10.5' },
      { to: '/history', label: 'History', icon: 'M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z' },
    ]
  },
]

function SidebarIcon({ d, className = 'w-[18px] h-[18px]' }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d={d} />
    </svg>
  )
}

function App() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <ErrorBoundary>
    <ToastProvider>
    <ConfirmProvider>
    <div className="min-h-screen flex bg-[var(--bg-primary)] text-[var(--text-primary)]">

      {/* ── Left Sidebar ─────────────────────────────────────── */}
      <aside className={`
        fixed top-0 left-0 h-screen z-40 flex flex-col
        bg-[#0c0d12] border-r border-slate-800/60
        transition-[width] duration-200 ease-in-out
        ${collapsed ? 'w-[60px]' : 'w-[220px]'}
      `}>

        {/* Logo */}
        <div className="flex items-center gap-2.5 px-4 h-14 shrink-0 border-b border-slate-800/40">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
            </svg>
          </div>
          {!collapsed && <span className="text-sm font-bold text-slate-200 whitespace-nowrap">SarabCraft</span>}
        </div>

        {/* Navigation Groups */}
        <nav className="flex-1 overflow-y-auto overflow-x-hidden py-3 px-2 space-y-4 scrollbar-none">
          {NAV_GROUPS.map(group => (
            <div key={group.label}>
              {!collapsed && (
                <div className="px-2 mb-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">
                    {group.label}
                  </span>
                </div>
              )}
              <div className="space-y-0.5">
                {group.items.map(item => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end || !item.matchPrefix}
                    className={({ isActive }) => `
                      group flex items-center gap-2.5 rounded-lg transition-all duration-150
                      ${collapsed ? 'justify-center px-0 py-2.5 mx-auto w-10' : 'px-2.5 py-2'}
                      ${isActive
                        ? 'bg-[var(--accent)]/10 text-[var(--accent)]'
                        : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/40'
                      }
                    `}
                    title={collapsed ? item.label : undefined}
                  >
                    {({ isActive }) => (
                      <>
                        <SidebarIcon d={item.icon} className={`w-[18px] h-[18px] shrink-0 transition-colors ${isActive ? 'text-[var(--accent)]' : 'text-slate-500 group-hover:text-slate-300'}`} />
                        {!collapsed && (
                          <span className={`text-[13px] font-medium whitespace-nowrap ${isActive ? 'text-[var(--accent)]' : ''}`}>
                            {item.label}
                          </span>
                        )}
                        {!collapsed && isActive && (
                          <div className="ml-auto w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
                        )}
                      </>
                    )}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* Bottom Section */}
        <div className="shrink-0 border-t border-slate-800/40 p-2 space-y-0.5">
          <NavLink
            to="/settings"
            className={({ isActive }) => `
              group flex items-center gap-2.5 rounded-lg transition-all duration-150
              ${collapsed ? 'justify-center px-0 py-2.5 mx-auto w-10' : 'px-2.5 py-2'}
              ${isActive
                ? 'bg-[var(--accent)]/10 text-[var(--accent)]'
                : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/40'
              }
            `}
            title={collapsed ? 'Settings' : undefined}
          >
            <svg className="w-[18px] h-[18px] shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            {!collapsed && <span className="text-[13px] font-medium">Settings</span>}
          </NavLink>

          <a
            href="/api/docs"
            target="_blank"
            rel="noopener noreferrer"
            className={`
              group flex items-center gap-2.5 rounded-lg transition-all duration-150
              text-slate-600 hover:text-slate-300 hover:bg-slate-800/40
              ${collapsed ? 'justify-center px-0 py-2.5 mx-auto w-10' : 'px-2.5 py-2'}
            `}
            title={collapsed ? 'API Docs' : undefined}
          >
            <svg className="w-[18px] h-[18px] shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
            </svg>
            {!collapsed && <span className="text-[13px] font-medium">API Docs</span>}
          </a>

          {/* Collapse Toggle */}
          <button
            onClick={() => setCollapsed(c => !c)}
            className={`
              group flex items-center gap-2.5 rounded-lg transition-all duration-150 w-full
              text-slate-600 hover:text-slate-300 hover:bg-slate-800/40
              ${collapsed ? 'justify-center px-0 py-2.5' : 'px-2.5 py-2'}
            `}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg className={`w-[18px] h-[18px] shrink-0 transition-transform duration-200 ${collapsed ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5" />
            </svg>
            {!collapsed && <span className="text-[13px] font-medium">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────── */}
      <div className={`flex-1 transition-[margin] duration-200 ease-in-out ${collapsed ? 'ml-[60px]' : 'ml-[220px]'}`}>
        {/* Thin top bar with breadcrumb */}
        <header className="h-11 border-b border-slate-800/40 bg-[var(--bg-primary)] sticky top-0 z-30 flex items-center px-6">
          <Breadcrumb path={location.pathname} />
        </header>

        <main className="px-6 py-6 max-w-7xl">
          <ImageAttackProvider>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/classifier" element={<ClassifierTab />} />
            <Route path="/image-attack" element={<ImageAttackTab />} />
            <Route path="/text-attack" element={<TextAttackTab />} />
            <Route path="/audio" element={<AudioTabs />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/batch" element={<BatchAttackPage />} />
            <Route path="/jobs" element={<JobsPage />} />
            <Route path="/robustness" element={<RobustnessHub />} />
            <Route path="/robustness/model" element={<RobustnessHub />} />
            <Route path="/robustness/attack" element={<RobustnessHub />} />
            <Route path="/settings" element={<Navigate to="/settings/credentials" replace />} />
            <Route path="/settings/:tab" element={<SettingsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
          </ImageAttackProvider>
        </main>
      </div>

    </div>
    </ConfirmProvider>
    </ToastProvider>
    </ErrorBoundary>
  )
}


const BREADCRUMB_MAP = {
  '/': 'Home',
  '/classifier': 'Analysis / Classify',
  '/image-attack': 'Image Attack',
  '/text-attack': 'Text Attack',
  '/audio': 'Audio Attack',
  '/dashboard': 'Analytics Dashboard',
  '/history': 'Analysis / History',
  '/batch': 'Batch Attack',
  '/jobs': 'Analysis / Jobs',
  '/robustness': 'Analysis / Robustness',
  '/robustness/model': 'Analysis / Robustness / Model',
  '/robustness/attack': 'Analysis / Robustness / Attack',
  '/settings/credentials': 'Settings / Credentials',
  '/settings/variables': 'Settings / Variables',
  '/settings/models': 'Settings / Models',
  '/settings/plugins': 'Settings / Plugins',
  '/settings/docs': 'Settings / Documentation',
}

function Breadcrumb({ path }) {
  const label = BREADCRUMB_MAP[path] || path.replace(/^\//, '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Home'
  const parts = label.split(' / ')

  return (
    <div className="flex items-center gap-1.5 text-[12px]">
      {parts.map((part, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {i > 0 && (
            <svg className="w-3 h-3 text-slate-700" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
            </svg>
          )}
          <span className={i === parts.length - 1 ? 'text-slate-300 font-medium' : 'text-slate-600'}>
            {part}
          </span>
        </span>
      ))}
    </div>
  )
}

export default App
