import type { MetricsResponse } from '../types/chat'

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await fetch('/metrics')
  if (!res.ok) {
    throw new Error(`Metrics API error: ${res.status}`)
  }
  return res.json()
}
