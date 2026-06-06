import type { FeedbackRequest, FeedbackResponse } from '../types/chat'

export async function sendFeedback(req: FeedbackRequest): Promise<FeedbackResponse> {
  const res = await fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    throw new Error(`Feedback API error: ${res.status}`)
  }
  return res.json()
}
