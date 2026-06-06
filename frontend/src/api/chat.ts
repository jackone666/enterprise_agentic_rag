const BASE = '/chat'

import type { ChatRequest, ChatResponse } from '../types/chat'

export async function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    throw new Error(`Chat API error: ${res.status} ${res.statusText}`)
  }
  return res.json()
}
