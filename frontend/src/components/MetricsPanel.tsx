import { useEffect, useState } from 'react'
import { BarChart3, RefreshCw } from './icons'
import { getMetrics } from '../api/metrics'
import type { MetricsSnapshot } from '../types/chat'

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true)
    setError('')
    try {
      const data = await getMetrics()
      setMetrics(data.metrics)
    } catch {
      setError('无法获取指标')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  if (!metrics && !loading && !error) return null

  return (
    <div className="p-4 text-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5 font-semibold text-gray-700">
          <BarChart3 size={16} />
          <span>运行指标</span>
        </div>
        <button onClick={load} disabled={loading} className="text-gray-400 hover:text-gray-600 disabled:opacity-50">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {error && <div className="text-xs text-red-500 mb-2">{error}</div>}

      {metrics && (
        <div className="space-y-2">
          {/* Stat cards */}
          <div className="grid grid-cols-2 gap-1.5">
            <StatCard label="总请求" value={metrics.total_requests} />
            <StatCard label="成功率" value={`${(metrics.success_rate * 100).toFixed(1)}%`} />
            <StatCard label="平均延迟" value={`${metrics.avg_latency_ms?.toFixed(1)}ms`} />
            <StatCard label="检索命中" value={`${(metrics.retrieval_hit_rate * 100).toFixed(1)}%`} />
            <StatCard label="校验通过" value={`${(metrics.verification_pass_rate * 100).toFixed(1)}%`} />
            <StatCard label="工具成功" value={`${(metrics.tool_success_rate * 100).toFixed(1)}%`} />
            <StatCard label="兜底率" value={`${(metrics.fallback_rate * 100).toFixed(1)}%`} color="amber" />
            <StatCard label="人工升级" value={`${(metrics.human_fallback_rate * 100).toFixed(1)}%`} color="red" />
          </div>

          {/* Intent distribution */}
          {metrics.intent_distribution && Object.keys(metrics.intent_distribution).length > 0 && (
            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">意图分布</div>
              <div className="space-y-0.5">
                {Object.entries(metrics.intent_distribution).sort(([, a], [, b]) => b - a).map(([intent, count]) => (
                  <div key={intent} className="flex justify-between text-xs">
                    <span className="text-gray-600">{intent}</span>
                    <span className="font-mono text-gray-800">{count as number}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-[10px] text-gray-400 pt-1 border-t border-gray-100">
            运行时长: {metrics.uptime_seconds?.toFixed(0) ?? 0}s
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  const colors: Record<string, string> = {
    amber: 'bg-amber-50 border-amber-200 text-amber-800',
    red: 'bg-red-50 border-red-200 text-red-800',
    default: 'bg-gray-50 border-gray-200 text-gray-800',
  }
  const c = colors[color ?? 'default'] ?? colors.default
  return (
    <div className={`rounded-lg border px-2.5 py-1.5 ${c}`}>
      <div className="text-[10px] opacity-70">{label}</div>
      <div className="font-mono font-semibold text-sm">{value}</div>
    </div>
  )
}
