import { NextResponse } from 'next/server'

const API_BASE = process.env.NG_PROXY_BASE || 'http://127.0.0.1:8099'

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))
  const action = body?.action || 'export_demo'

  if (action === 'export_demo') {
    const resp = await fetch(`${API_BASE}/export_demo`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({}),
    })
    const data = await resp.json()
    return NextResponse.json(data)
  }

  if (action === 'publish') {
    const payload = body?.payload || {}
    const resp = await fetch(`${API_BASE}/publish`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await resp.json()
    return NextResponse.json(data)
  }

  if (action === 'publish_multi') {
    const payload = body?.payload || {}
    const resp = await fetch(`${API_BASE}/publish_multi`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await resp.json()
    return NextResponse.json(data)
  }

  return NextResponse.json({ error: 'unknown action' }, { status: 400 })
}


