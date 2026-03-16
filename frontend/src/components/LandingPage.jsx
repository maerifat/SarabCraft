import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSystemInfo, getHistoryStats } from '../api/client'

export default function LandingPage() {
  const navigate = useNavigate()
  const [sysInfo, setSysInfo] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getSystemInfo().then(setSysInfo).catch(() => {})
    getHistoryStats().then(setStats).catch(() => {})
  }, [])

  return (
    <div className="space-y-16 pb-12">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900 via-slate-800/80 to-slate-900">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-cyan-500/8 via-transparent to-transparent" />
        <div className="relative px-8 py-20 text-center">
          {sysInfo && (
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700/60 mb-8">
              <span className={`w-1.5 h-1.5 rounded-full ${sysInfo.cuda_available ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              <span className="text-[11px] text-slate-500">
                {sysInfo.cuda_available ? sysInfo.gpu_name : 'CPU mode'} - {sysInfo.image_attacks} image attacks, {sysInfo.image_models} image models, {(sysInfo.audio_models || 0) + (sysInfo.asr_models || 0)} audio/ASR models
              </span>
            </div>
          )}
          <h1 className="text-4xl md:text-5xl font-black tracking-tight text-white mb-4">
            SarabCraft
          </h1>
          <p className="text-sm md:text-base font-semibold text-cyan-300 max-w-2xl mx-auto mb-3">
            The Multimodal Adversarial AI Security Framework.
          </p>
          <p className="text-base text-slate-400 max-w-3xl mx-auto mb-10 leading-relaxed">
            SarabCraft helps security teams and researchers craft adversarial images and audio, validate
            transfer across local and remote targets, onboard custom models from Settings, and keep
            long-running experiments organized with jobs, artifacts, history, and reports.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <button onClick={() => navigate('/image-attack')}
              className="px-7 py-3 bg-white text-slate-900 font-bold rounded-lg hover:bg-slate-100 transition-all text-sm">
              Image Attacks
            </button>
            <button onClick={() => navigate('/audio')}
              className="px-7 py-3 bg-slate-800 border border-slate-600 text-slate-200 font-semibold rounded-lg hover:bg-slate-700 transition-all text-sm">
              Audio Attacks
            </button>
            <button onClick={() => navigate('/settings/models')}
              className="px-7 py-3 bg-slate-800 border border-slate-700 text-slate-300 font-medium rounded-lg hover:text-slate-100 hover:border-slate-600 transition-all text-sm">
              Manage Models
            </button>
            <button onClick={() => navigate('/jobs')}
              className="px-7 py-3 bg-slate-800/60 border border-slate-700 text-slate-400 font-medium rounded-lg hover:text-slate-200 hover:border-slate-600 transition-all text-sm">
              Jobs Queue
            </button>
          </div>
          <p className="text-xs text-slate-500 max-w-2xl mx-auto mt-4 leading-relaxed">
            Start in Settings &gt; Models to register custom local checkpoints or remote verification targets,
            then move into image attacks, audio attacks, jobs, dashboard, and history.
          </p>
        </div>
      </section>

      {/* What it does — not how it's built */}
      <section className="grid md:grid-cols-2 gap-5">
        <CapabilityCard
          title="Image Attack Research"
          items={[
            '32+ image attacks across gradient, optimization, black-box, physical, and research workflows',
            'Targeted and transfer-oriented studies, including SarabCraft R1',
            'Run against built-in or custom local image models from Settings > Models',
            'Capture perturbation metrics and visual outputs for every run',
          ]}
          action="Run Image Attack"
          onClick={() => navigate('/image-attack')}
        />
        <CapabilityCard
          title="Audio Attack Research"
          items={[
            'Targeted transcription, hidden command, universal muting, psychoacoustic, over-the-air, speech jamming, and UA3 workflows',
            'Support for audio classification and ASR attack workflows in one place',
            'Use built-in or custom audio and ASR models registered from Settings > Models',
            'Validate remote audio transfer against supported speech-to-text targets',
          ]}
          action="Run Audio Attack"
          onClick={() => navigate('/audio')}
        />
        <CapabilityCard
          title="Model & Target Management"
          items={[
            'Add custom image, audio, and ASR models directly from Settings > Models',
            'Register Hugging Face API and cloud verification targets without editing config.py',
            'Enable compatibility flags so one model can serve classify, attack, robustness, benchmark, and local verification',
            'Test, duplicate, disable, or archive entries from the UI',
          ]}
          action="Manage Models"
          onClick={() => navigate('/settings/models')}
        />
        <CapabilityCard
          title="Jobs, Analysis & Reporting"
          items={[
            'Run long attacks, robustness sweeps, and benchmarks through durable jobs',
            'Track progress, artifacts, recent events, cancellation, and resume support in Jobs',
            'Compare outcomes in Dashboard and History with transferability and success-rate views',
            'Export HTML and JSON evidence for write-ups, demos, and client work',
          ]}
          action="Open Jobs"
          onClick={() => navigate('/jobs')}
        />
      </section>

      {/* Attack surface */}
      <section>
        <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-6">Image Attack Coverage</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {[
            { name: 'Gradient-Based', count: 14, color: 'border-cyan-500/30 bg-cyan-500/5' },
            { name: 'Optimization', count: 6, color: 'border-purple-500/30 bg-purple-500/5' },
            { name: 'Transfer', count: 5, color: 'border-emerald-500/30 bg-emerald-500/5' },
            { name: 'Black-Box', count: 5, color: 'border-orange-500/30 bg-orange-500/5' },
            { name: 'Physical', count: 2, color: 'border-red-500/30 bg-red-500/5' },
            { name: 'Research', count: 4, color: 'border-amber-500/30 bg-amber-500/5' },
          ].map(c => (
            <div key={c.name} className={`rounded-xl p-4 border text-center ${c.color}`}>
              <div className="text-2xl font-black text-white">{c.count}</div>
              <div className="text-[10px] text-slate-500 mt-1 uppercase tracking-wider">{c.name}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Verification targets */}
      <section>
        <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-6">Transfer Targets</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { name: 'AWS Rekognition', type: 'Remote image verification' },
            { name: 'Azure Computer Vision', type: 'Remote image verification' },
            { name: 'Google Cloud Vision', type: 'Remote image verification' },
            { name: 'Hugging Face API Targets', type: 'Remote image verification' },
            { name: 'AWS Transcribe', type: 'Remote audio verification' },
            { name: 'ElevenLabs STT', type: 'Remote audio verification' },
            { name: 'Local Image Models', type: 'Registry-backed local verification' },
            { name: 'Custom Plugins', type: 'User-defined image or audio targets' },
          ].map(t => (
            <div key={t.name} className="bg-slate-800/30 border border-slate-700/40 rounded-lg px-4 py-3">
              <div className="text-sm text-slate-300 font-medium">{t.name}</div>
              <div className="text-[10px] text-slate-600 mt-0.5">{t.type}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Session stats — only if there's data */}
      {stats && stats.total > 0 && (
        <section>
          <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-6">Research History</h2>
          <div className="grid grid-cols-3 gap-4">
            <StatBlock value={stats.total} label="Runs Recorded" />
            <StatBlock value={`${(stats.success_rate * 100).toFixed(1)}%`} label="Success Rate"
              color={stats.success_rate > 0.5 ? 'text-emerald-400' : 'text-amber-400'} />
            <StatBlock value={Object.keys(stats.models || {}).length} label="Models Covered" />
          </div>
        </section>
      )}

      {/* Minimal footer */}
      <footer className="text-center text-[11px] text-slate-700 pt-8 border-t border-slate-800/50">
        SarabCraft
        <span className="mx-2">·</span>
        <a href="/api/docs" target="_blank" rel="noopener noreferrer" className="text-slate-600 hover:text-slate-400 transition">API Documentation</a>
        <span className="mx-2">·</span>
        <a href="/api/redoc" target="_blank" rel="noopener noreferrer" className="text-slate-600 hover:text-slate-400 transition">OpenAPI Reference</a>
      </footer>
    </div>
  )
}


function CapabilityCard({ title, items, action, onClick }) {
  return (
    <div className="bg-slate-800/25 border border-slate-700/40 rounded-xl p-6 flex flex-col">
      <h3 className="text-sm font-bold text-slate-200 mb-4">{title}</h3>
      <ul className="space-y-2.5 flex-1">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2.5 text-xs text-slate-400 leading-relaxed">
            <span className="text-slate-600 mt-0.5 shrink-0">—</span>
            {item}
          </li>
        ))}
      </ul>
      <button onClick={onClick}
        className="mt-5 text-xs text-slate-500 hover:text-slate-200 transition font-medium self-start">
        {action} →
      </button>
    </div>
  )
}


function StatBlock({ value, label, color = 'text-white' }) {
  return (
    <div className="bg-slate-800/30 border border-slate-700/40 rounded-xl p-5 text-center">
      <div className={`text-2xl font-black ${color}`}>{value}</div>
      <div className="text-[10px] text-slate-600 mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}
