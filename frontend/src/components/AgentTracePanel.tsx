import type { ChatResponse, NodeEvent } from '../types/chat'
import { CheckCircle, XCircle, SkipForward, Activity } from './icons'

const ALL_NODES = [
  'load_memory', 'check_permission', 'deep_intent_recognition', 'master_agent',
  'retrieve_knowledge', 'call_tools', 'rewrite_query', 'build_context',
  'generate_code', 'execute_code', 'generate_answer', 'verify_answer', 'finalize_answer',
  'save_memory',
]

const NODE_LABELS: Record<string, string> = {
  load_memory: '加载记忆',
  check_permission: '权限检查',
  deep_intent_recognition: '深度意图',
  master_agent: '主 Agent 调度',
  retrieve_knowledge: '知识检索',
  call_tools: '工具调用',
  rewrite_query: '查询改写',
  build_context: '上下文构建',
  generate_code: '代码生成',
  execute_code: '代码执行',
  generate_answer: '生成答案',
  verify_answer: '答案校验',
  finalize_answer: '答案定稿',
  save_memory: '保存记忆',
}

export default function AgentTracePanel({ response }: { response: ChatResponse | null }) {
  if (!response) {
    return (
      <div className="p-4 text-sm text-gray-400 text-center">
        <Activity size={24} className="mx-auto mb-2 opacity-50" />
        <p>等待请求...</p>
        <p className="text-xs mt-1">发送问题后此处展示 Agent 执行流程</p>
      </div>
    )
  }

  const nodeEvents: NodeEvent[] = response.node_events ?? []
  const nodeEndMap = new Map<string, NodeEvent>()
  for (const evt of nodeEvents) {
    if (evt.event_type === 'node_end') {
      nodeEndMap.set(evt.node_name, evt)
    }
  }

  return (
    <div className="p-4 space-y-3 text-sm">
      {/* Trace summary */}
      <div className="rounded-lg bg-gray-50 border border-gray-200 p-3 space-y-1.5 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-500">Trace</span>
          <span className="font-mono text-gray-700">{response.trace_id?.slice(0, 12) ?? '-'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Intent</span>
          <span className="text-purple-700 font-medium">{response.intent ?? '-'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Verified</span>
          <span className={response.verified ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
            {response.verified ? '通过' : '未通过'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">检索后端</span>
          <span className="text-blue-700 font-medium">{response.retrieval_backend ?? response.pipeline_trace?.backend ?? '-'}</span>
        </div>
        {response.pipeline_trace && (
          <div className="flex justify-between">
            <span className="text-gray-500">总延迟</span>
            <span className="text-gray-700">{response.pipeline_trace.total_latency_ms?.toFixed(1)}ms</span>
          </div>
        )}
      </div>

      {/* Eval scores */}
      {response.eval_result && (
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 space-y-1 text-xs">
          <div className="font-semibold text-blue-800 mb-1">📊 评估分数</div>
          <div className="grid grid-cols-2 gap-1">
            <EvalBadge label="综合" value={response.eval_result.overall} highlight />
            <EvalBadge label="精确度" value={response.eval_result.precision} />
            <EvalBadge label="召回率" value={response.eval_result.recall} />
            <EvalBadge label="忠实度" value={response.eval_result.faithfulness} />
            <EvalBadge label="相关性" value={response.eval_result.relevance} />
          </div>
          <div className={`text-[10px] font-medium mt-1 ${response.eval_result.passing ? 'text-green-600' : 'text-red-600'}`}>
            {response.eval_result.passing ? '✅ 评估通过' : '❌ 评估未通过'}
          </div>
        </div>
      )}

      {/* Flow visualization */}
      <div>
        <div className="text-xs font-semibold text-gray-600 mb-2 flex items-center gap-1.5">
          <Activity size={13} /> Agent 执行流程
        </div>
        <div className="space-y-0">
          {ALL_NODES.map((nodeName, i) => {
            const evt = nodeEndMap.get(nodeName)
            let status: 'success' | 'failed' | 'skipped'
            let Icon
            let colorClass

            if (!evt) {
              // Check if this node COULD have run (based on route)
              status = 'skipped'
              Icon = SkipForward
              colorClass = 'text-gray-300'
            } else if (evt.success) {
              status = 'success'
              Icon = CheckCircle
              colorClass = 'text-green-500'
            } else {
              status = 'failed'
              Icon = XCircle
              colorClass = 'text-red-500'
            }

            return (
              <div key={nodeName} className="flex gap-2 items-start py-1">
                {/* Connector line + dot */}
                <div className="flex flex-col items-center shrink-0 pt-0.5">
                  <Icon size={14} className={colorClass} />
                  {i < ALL_NODES.length - 1 && (
                    <div className={`w-px h-5 ${evt ? 'bg-gray-300' : 'bg-gray-200'}`} />
                  )}
                </div>
                <div className="flex-1 min-w-0 pb-1.5">
                  <div className="flex items-center justify-between">
                    <span className={`font-mono text-xs ${evt ? 'text-gray-800' : 'text-gray-400'}`}>
                      {NODE_LABELS[nodeName] ?? nodeName}
                    </span>
                    <span className="text-[10px] text-gray-400">
                      {evt ? `${evt.latency_ms?.toFixed(1)}ms` : status === 'skipped' ? '—' : ''}
                    </span>
                  </div>
                  <div className="text-[10px] text-gray-400 font-mono">{nodeName}</div>
                  {status === 'skipped' && (
                    <div className="text-[10px] text-gray-300">跳过</div>
                  )}
                  {evt?.error && (
                    <div className="text-[10px] text-red-500 truncate">{evt.error}</div>
                  )}
                  {evt?.output_summary && status === 'success' && (
                    <div className="text-[10px] text-gray-400 truncate">{evt.output_summary.slice(0, 60)}</div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-3 text-[10px] border-t border-gray-100 pt-2">
        <span className="flex items-center gap-1"><CheckCircle size={10} className="text-green-500" /> 成功</span>
        <span className="flex items-center gap-1"><XCircle size={10} className="text-red-500" /> 失败</span>
        <span className="flex items-center gap-1"><SkipForward size={10} className="text-gray-300" /> 跳过</span>
      </div>
    </div>
  )
}

function EvalBadge({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  const pct = (value * 100).toFixed(0)
  const color = value >= 0.7 ? 'text-green-600' : value >= 0.4 ? 'text-amber-600' : 'text-red-600'
  return (
    <div className="flex justify-between items-center">
      <span className="text-gray-500">{label}</span>
      <span className={`font-mono font-medium ${highlight ? 'text-blue-700 text-sm' : color}`}>
        {pct}%
      </span>
    </div>
  )
}
