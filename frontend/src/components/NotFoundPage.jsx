import { NavLink } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center px-4">
      <div className="text-center animate-fade-in">
        <p className="text-8xl font-black text-slate-700 select-none mb-2 tracking-tight not-found-glow">
          404
        </p>
        <h1 className="text-xl font-semibold text-slate-200 mb-2">Page not found</h1>
        <p className="text-slate-400 mb-8 max-w-sm mx-auto leading-relaxed">
          The page you are looking for does not exist or has been moved.
        </p>
        <NavLink
          to="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25" />
          </svg>
          Go Home
        </NavLink>
      </div>
    </div>
  )
}
