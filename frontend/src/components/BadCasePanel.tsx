import { useEffect, useState } from 'react'
import { AlertTriangle, Search, ShieldAlert, RefreshCw } from './icons'

interface BadCase {
  trace_id: string
  query: string
  reason: string
  source: string
  created_at: string
}

interface Alert {
  level: string
  msg: string
}

export default function BadCasePanel() {
  const [cases, setCases] = useState<BadCase[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [stats, setStats] = useState<Record<string, unknown>>({})
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(false)

  async function load(source = '') {
    setLoading(true)
    try {
      const url = `/admin/bad-cases?limit=100${source ? `&source=${source}` : ''}`
      const r = await fetch(url)
      const data = await r.json()
      setCases(data.bad_cases || [])
      setAlerts(data.alerts || [])
      setStats(data.stats || {})
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const sourceDist = (stats.source_distribution || {}) as Record<string, number>

  return (
    <div className="p-4 text-sm space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 font-semibold text-gray-700">
          <ShieldAlert size={14} /> 告警与坏例监控
        </div>
        <button onClick={() => load(filter)} disabled={loading} className="text-gray-400 hover:text-gray-600">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-1">
          {alerts.map((a, i) => (
            <div key={i} className={`flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs font-medium ${
              a.level === 'error' ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}>
              <AlertTriangle size={12} />
              {a.msg}
            </div>
          ))}
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 gap-1.5">
        <div className="rounded-lg bg-gray-50 border px-2.5 py-1.5">
          <div className="text-[10px] text-gray-500">坏例总数</div>
          <div className="font-mono font-semibold text-sm">{String(stats.total ?? 0)}</div>
        </div>
        {Object.entries(sourceDist).map(([k, v]) => (
          <div key={k} className="rounded-lg bg-gray-50 border px-2.5 py-1.5">
            <div className="text-[10px] text-gray-500">{k}</div>
            <div className="font-mono font-semibold text-sm">{v as number}</div>
          </div>
        ))}
      </div>

      {/* Filter */}
      <div className="flex gap-1.5">
        <select
          value={filter}
          onChange={e => { setFilter(e.target.value); load(e.target.value) }}
          className="flex-1 rounded border border-gray-200 px-2 py-1 text-xs"
        >
          <option value="">全部来源</option>
          <option value="auto">自动捕获</option>
          <option value="feedback">用户反馈</option>
          <option value="regression">回归测试</option>
          <option value="eval">评估系统</option>
        </select>
        <div className="flex items-center gap-1 text-[10px] text-gray-400">
          <Search size={12} />
          {cases.length} 条
        </div>
      </div>

      {/* Case list */}
      <div className="space-y-1 max-h-[500px] overflow-y-auto">
        {cases.length === 0 && (
          <div className="text-xs text-gray-400 text-center py-4">暂无坏例</div>
        )}
        {cases.map((c, i) => (
          <div key={i} className="rounded-lg border border-gray-200 bg-white p-2 text-xs space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-medium text-gray-800 truncate max-w-[200px]">{c.query || '(无问题)'}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                c.source === 'feedback' ? 'bg-red-50 text-red-600' :
                c.source === 'auto' ? 'bg-amber-50 text-amber-600' :
                c.source === 'eval' ? 'bg-purple-50 text-purple-600' :
                'bg-blue-50 text-blue-600'
              }`}>{c.source}</span>
            </div>
            <div className="text-gray-500 truncate">{c.reason || '-'}</div>
            <div className="flex justify-between text-[10px] text-gray-400">
              <span className="font-mono">{c.trace_id?.slice(0, 12) || '-'}</span>
              <span>{c.created_at?.slice(0, 19) || '-'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
