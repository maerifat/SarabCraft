export function downloadB64(b64, filename, mimeType = 'image/png') {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  const blob = new Blob([bytes], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function downloadBundle(files) {
  files.forEach(({ b64, filename, mime }) => {
    setTimeout(() => downloadB64(b64, filename, mime), 0)
  })
}

export function slugify(s) {
  return (s || 'unknown').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 40)
}
