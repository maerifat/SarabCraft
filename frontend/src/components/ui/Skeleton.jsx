export function Skeleton({ className = '', count = 1 }) {
  return Array.from({ length: count }, (_, i) => (
    <div key={i} className={`animate-pulse bg-slate-800/60 rounded-lg ${className}`} />
  ))
}

export function CardSkeleton() {
  return (
    <div className="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 space-y-4 animate-pulse">
      <div className="h-4 bg-slate-700/60 rounded w-1/3" />
      <div className="h-3 bg-slate-700/40 rounded w-2/3" />
      <div className="grid grid-cols-3 gap-3 mt-4">
        <div className="h-20 bg-slate-700/40 rounded-lg" />
        <div className="h-20 bg-slate-700/40 rounded-lg" />
        <div className="h-20 bg-slate-700/40 rounded-lg" />
      </div>
    </div>
  )
}

export function TableSkeleton({ rows = 5, cols = 6 }) {
  return (
    <div className="space-y-2 animate-pulse">
      <div className="flex gap-4 pb-2 border-b border-slate-700/50">
        {Array.from({ length: cols }, (_, i) => (
          <div key={i} className="h-3 bg-slate-700/60 rounded flex-1" />
        ))}
      </div>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex gap-4 py-2">
          {Array.from({ length: cols }, (_, j) => (
            <div key={j} className="h-3 bg-slate-700/30 rounded flex-1" />
          ))}
        </div>
      ))}
    </div>
  )
}
