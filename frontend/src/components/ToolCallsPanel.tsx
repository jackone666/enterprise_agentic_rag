import type { ChatResponse, ToolResult } from '../types/chat'
import { Wrench } from './icons'

export default function ToolCallsPanel({ response }: { response: ChatResponse | null }) {
  if (!response) {
    return (
      <div className="p-4 text-sm text-gray-400 text-center">
        <Wrench size={24} className="mx-auto mb-2 opacity-50" />
        <p>等待请求...</p>
      </div>
    )
  }

  const tools = response.tool_results ?? []
  const errors = response.tool_errors ?? []

  return (
    <div className="p-4 space-y-3 text-sm">
      <div className="flex items-center gap-1.5 font-semibold text-gray-700">
        <Wrench size={14} /> 工具调用
      </div>

      {tools.length === 0 && errors.length === 0 ? (
        <div className="text-xs text-gray-400">当前请求未触发工具调用</div>
      ) : (
        <div className="space-y-2">
          {tools.map((tr, i) => <ToolCard key={i} tr={tr} />)}
        </div>
      )}

      {errors.length > 0 && (
        <div className="mt-2 rounded-lg border border-red-200 bg-red-50 p-2.5">
          <div className="text-xs font-medium text-red-700">工具错误 ({errors.length})</div>
          {errors.map((e, i) => (
            <div key={i} className="text-[10px] text-red-600 mt-0.5 font-mono">{e}</div>
          ))}
        </div>
      )}

      {/* Retry info */}
      {response.retry_count && Object.keys(response.retry_count).length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <div className="text-[10px] text-gray-500">重试计数</div>
          {Object.entries(response.retry_count).map(([k, v]) => (
            <div key={k} className="text-[10px] font-mono text-gray-600">{k}: {v as number}次</div>
          ))}
        </div>
      )}
    </div>
  )
}

function ToolCard({ tr }: { tr: ToolResult }) {
  const outputStr = typeof tr.output === 'string' ? tr.output : JSON.stringify(tr.output)
  return (
    <div className={`rounded-lg border p-2.5 ${
      tr.success ? 'border-green-200 bg-green-50/50' : 'border-red-200 bg-red-50/50'
    }`}>
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-xs font-medium">{tr.tool_name}</span>
        <span className="text-[10px] text-gray-400">{tr.latency_ms?.toFixed(1) ?? '?'}ms</span>
      </div>
      <div className="flex items-center gap-1 text-[10px]">
        <span className={tr.success ? 'text-green-600' : 'text-red-600'}>{tr.success ? '✅ 成功' : '❌ 失败'}</span>
      </div>
      {tr.error && <div className="text-[10px] text-red-600 mt-0.5">{tr.error}</div>}
      {tr.success && outputStr && (
        <div className="text-[10px] text-gray-600 mt-1 line-clamp-3 font-mono bg-white/50 rounded p-1">{outputStr.slice(0, 200)}</div>
      )}
    </div>
  )
}
