function escapeHtml(str) {
  if (str == null) return ''
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

export function generateHTMLReport(result, attack, model, metrics, transferResults) {
  const timestamp = escapeHtml(new Date().toISOString())

  const metricsRows = metrics ? Object.entries(metrics).map(([k, v]) =>
    `<tr><td>${escapeHtml(k.replace(/_/g, ' ').toUpperCase())}</td><td>${typeof v === 'number' ? escapeHtml(v.toFixed(6)) : escapeHtml(v)}</td></tr>`
  ).join('') : ''

  const transferRows = (transferResults || []).map(r =>
    `<tr><td>${escapeHtml(r.service || r.name || '—')}</td><td>${r.matched_target ? 'TARGET MATCHED' : r.original_label_gone ? 'ORIGINAL GONE' : 'NO EFFECT'}</td><td>${escapeHtml(r.elapsed_ms || '—')}ms</td></tr>`
  ).join('')

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SarabCraft — Attack Report</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;color:#e2e8f0;padding:2rem}
.container{max-width:900px;margin:0 auto}
h1{font-size:1.5rem;font-weight:800;margin-bottom:0.5rem;color:#fff}
h2{font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.15em;color:#64748b;margin:2rem 0 1rem;border-bottom:1px solid #1e293b;padding-bottom:0.5rem}
.subtitle{font-size:0.8rem;color:#94a3b8;margin-bottom:2rem}
.badge{display:inline-block;padding:0.25rem 0.75rem;border-radius:9999px;font-size:0.7rem;font-weight:600}
.badge-success{background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.2)}
.badge-fail{background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.2)}
.card{background:rgba(30,41,59,0.4);border:1px solid #334155;border-radius:0.75rem;padding:1.25rem;margin-bottom:1rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem}
table{width:100%;font-size:0.75rem;border-collapse:collapse}
th,td{text-align:left;padding:0.5rem 0.75rem;border-bottom:1px solid #1e293b}
th{color:#64748b;font-weight:600;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em}
td{color:#cbd5e1}
img{max-width:100%;border-radius:0.5rem;border:1px solid #334155}
.metric-card{text-align:center;background:rgba(30,41,59,0.6);border:1px solid #334155;border-radius:0.5rem;padding:0.75rem}
.metric-val{font-size:1rem;font-weight:800;color:#fff}
.metric-label{font-size:0.55rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.25rem}
.footer{text-align:center;margin-top:3rem;color:#475569;font-size:0.7rem}
</style>
</head>
<body>
<div class="container">
<h1>SarabCraft — Attack Report</h1>
<p class="subtitle">Generated ${timestamp}</p>

<div class="card">
<table>
<tr><th style="width:140px">Attack</th><td><strong>${escapeHtml(attack || '—')}</strong></td></tr>
<tr><th>Model</th><td>${escapeHtml(model || '—')}</td></tr>
<tr><th>Result</th><td><span class="badge ${result?.success ? 'badge-success' : 'badge-fail'}">${result?.success ? 'SUCCESS' : 'FAILED'}</span></td></tr>
<tr><th>Original Class</th><td>${escapeHtml(result?.original_class || '—')}</td></tr>
<tr><th>Adversarial Class</th><td>${escapeHtml(result?.adversarial_class || '—')}</td></tr>
<tr><th>Target Class</th><td>${escapeHtml(result?.target_class || '—')}</td></tr>
</table>
</div>

${metrics ? `
<h2>Perturbation Quality Metrics</h2>
<div class="card">
<table>${metricsRows}</table>
</div>` : ''}

${result?.adversarial_b64 || result?.perturbation_b64 ? `
<h2>Images</h2>
<div class="grid">
${result?.adversarial_b64 ? `<div><p style="font-size:0.65rem;color:#64748b;margin-bottom:0.5rem">ADVERSARIAL OUTPUT</p><img src="data:image/png;base64,${result.adversarial_b64}" alt="Adversarial"></div>` : ''}
${result?.perturbation_b64 ? `<div><p style="font-size:0.65rem;color:#64748b;margin-bottom:0.5rem">PERTURBATION (10x)</p><img src="data:image/png;base64,${result.perturbation_b64}" alt="Perturbation"></div>` : ''}
</div>` : ''}

${transferRows ? `
<h2>Transfer Verification Results</h2>
<div class="card">
<table>
<thead><tr><th>Service</th><th>Result</th><th>Time</th></tr></thead>
<tbody>${transferRows}</tbody>
</table>
</div>` : ''}

<div class="footer">
<p>SarabCraft v2.0 — Crafting illusions that machines believe.</p>
<p style="margin-top:0.25rem">Report generated for Black Hat Arsenal 2026</p>
</div>
</div>
</body>
</html>`

  return html
}


export function downloadReport(result, attack, model, metrics, transferResults) {
  const html = generateHTMLReport(result, attack, model, metrics, transferResults)
  const blob = new Blob([html], { type: 'text/html' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `sarabcraft_report_${new Date().toISOString().slice(0, 10)}_${(attack || 'attack').replace(/[^a-z0-9]/gi, '_')}.html`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}


export function downloadJSON(result, attack, model, metrics) {
  const data = {
    tool: 'SarabCraft v2.0',
    timestamp: new Date().toISOString(),
    attack,
    model,
    success: result?.success,
    original_class: result?.original_class,
    adversarial_class: result?.adversarial_class,
    target_class: result?.target_class,
    metrics,
    original_preds: result?.original_preds,
    adversarial_preds: result?.adversarial_preds,
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `sarabcraft_report_${new Date().toISOString().slice(0, 10)}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
