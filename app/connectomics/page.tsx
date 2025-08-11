"use client"
import { useState } from 'react'

export default function ConnectomicsPage() {
  const [busy, setBusy] = useState(false)
  const [viewerUrl, setViewerUrl] = useState<string | null>(null)
  const [imagePath, setImagePath] = useState<string | null>(null)
  const [segPath, setSegPath] = useState<string | null>(null)

  const runDemo = async () => {
    try {
      setBusy(true)
      setViewerUrl(null)
      const resp = await fetch('/api/connectomics', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ action: 'export_demo' }),
      })
      const data = await resp.json()
      setViewerUrl(data.viewer_url)
      setImagePath(data.image_path)
      setSegPath(data.seg_path)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ padding: 24, maxWidth: 800 }}>
      <h1>Connectomics MVP</h1>
      <p>Export a demo precomputed dataset and get a Neuroglancer viewer URL.</p>
      <button onClick={runDemo} disabled={busy} style={{ padding: '8px 16px' }}>
        {busy ? 'Working…' : 'Run Demo'}
      </button>

      {viewerUrl && (
        <div style={{ marginTop: 16 }}>
          <div>
            <strong>Viewer URL:</strong>
          </div>
          <div>
            <a href={viewerUrl} target="_blank" rel="noreferrer">
              Open Neuroglancer
            </a>
          </div>
        </div>
      )}

      {(imagePath || segPath) && (
        <div style={{ marginTop: 16 }}>
          <div>
            <strong>Local precomputed paths</strong>
          </div>
          {imagePath && <div>Image: {imagePath}</div>}
          {segPath && <div>Segmentation: {segPath}</div>}
        </div>
      )}
    </div>
  )
}


