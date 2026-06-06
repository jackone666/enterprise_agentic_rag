import type { ToolResult } from '../types/chat'

export default function ToolResultPanel({ results }: { results: ToolResult[] }) {
  if (!results || results.length === 0) return null

  return (
    <div className="mt-2">
      <div className="font-semibold text-xs text-gray-600 mb-1">🔧 工具执行结果</div>
      <div className="space-y-1">
        {results.map((tr, i) => (
          <div
            key={i}
            className={`text-xs rounded px-2 py-1.5 border ${
              tr.success
                ? 'bg-green-50 border-green-200 text-green-800'
                : 'bg-red-50 border-red-200 text-red-800'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium">
                {tr.success ? '✅' : '❌'} {tr.tool_name ?? 'unknown'}
              </span>
              <span className="text-gray-400">{tr.latency_ms?.toFixed(1) ?? '?'}ms</span>
            </div>
            {tr.error && !tr.success && (
              <div className="mt-0.5 text-red-600">{tr.error}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
