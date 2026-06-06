import { useEffect, useState } from 'react'
import type { MetricsSnapshot } from '../types/chat'

export default function MetricsMiniBar() {
  const [m, setM] = useState<MetricsSnapshot | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const res = await fetch('/metrics')
        const data = await res.json()
        if (alive) setM(data.metrics)
      } catch { /* ignore */ }
    }
    load()
    return () => { alive = false }
  }, [])

  if (!m) return null

  return (
    <div className="flex items-center gap-3 text-[10px] text-gray-500">
      <span className="font-mono">{m.total_requests ?? 0} 请求</span>
      <span className="text-green-600">命中 {(m.retrieval_hit_rate ?? 0) * 100 | 0}%</span>
      <span className="text-blue-600">校验 {(m.verification_pass_rate ?? 0) * 100 | 0}%</span>
      <span className="text-amber-600">兜底 {(m.fallback_rate ?? 0) * 100 | 0}%</span>
    </div>
  )
}
