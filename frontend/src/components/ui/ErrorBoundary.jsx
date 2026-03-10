import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, showDetails: false }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ErrorBoundary]', error, errorInfo)
  }

  handleReload = () => {
    window.location.reload()
  }

  toggleDetails = () => {
    this.setState((prev) => ({ showDetails: !prev.showDetails }))
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    const { error, showDetails } = this.state

    return (
      <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center px-4">
        <div className="max-w-lg w-full text-center">
          <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          </div>

          <h1 className="text-2xl font-bold text-slate-100 mb-2">Something went wrong</h1>
          <p className="text-slate-400 mb-8 leading-relaxed">
            An unexpected error occurred. You can try reloading the page to recover.
          </p>

          <div className="flex items-center justify-center gap-3 mb-8">
            <button
              onClick={this.handleReload}
              className="px-5 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition-colors"
            >
              Reload Page
            </button>
            <button
              onClick={this.toggleDetails}
              className="px-5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-sm font-medium transition-colors"
            >
              {showDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>

          {showDetails && error && (
            <div className="text-left bg-slate-900/80 border border-slate-700/50 rounded-lg p-4 overflow-auto max-h-64">
              <p className="text-red-400 text-sm font-mono mb-2">{error.toString()}</p>
              {error.stack && (
                <pre className="text-xs text-slate-500 font-mono whitespace-pre-wrap break-words">
                  {error.stack}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    )
  }
}
