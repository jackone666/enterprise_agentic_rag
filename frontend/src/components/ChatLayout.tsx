import { useState, useRef, useCallback, useEffect } from 'react'
import { MessageSquare, Activity, Database, Wrench, ShieldAlert, BarChart3, Upload } from './icons'
import { sendChatMessage } from '../api/chat'
import type { ChatMessage, ChatResponse } from '../types/chat'
import ChatMessageBubble from './ChatMessage'
import ChatInput from './ChatInput'
import AgentTracePanel from './AgentTracePanel'
import RAGEvidencePanel from './RAGEvidencePanel'
import ToolCallsPanel from './ToolCallsPanel'
import FallbackPanel from './FallbackPanel'
import MetricsMiniBar from './MetricsMiniBar'
import SystemStatusBadge from './SystemStatusBadge'
import FileUpload from './FileUpload'
import BadCasePanel from './BadCasePanel'

function genId() { return Math.random().toString(36).slice(2, 10) }

const CAPABILITY_TAGS = [
  'LangGraph', 'Multi-Agent', 'Agentic RAG', 'Tool Calling',
  'Answer Verification', 'Human Fallback', 'Observability', 'Evaluation Flywheel',
]

const DEMO_QUESTIONS = [
  { label: '排障类', q: '我接入 SDK 时遇到 AUTH_401 错误怎么办？', intent: 'troubleshooting' },
  { label: '技术类', q: 'API 认证方式有哪些？', intent: 'technical_question' },
  { label: '工单类', q: '查询工单 TKT-001 的状态', intent: 'ticket_query' },
  { label: '策略类', q: '数据分类标准是什么？有什么合规要求？', intent: 'policy_question' },
  { label: '系统类', q: '当前系统各服务状态正常吗？', intent: 'troubleshooting' },
  { label: '安全类', q: '没有权限的用户能查看机密数据吗？', intent: 'policy_question' },
]

export default function ChatLayout() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [activeResponse, setActiveResponse] = useState<ChatResponse | null>(null)
  const [error, setError] = useState('')
  const [rightTab, setRightTab] = useState<'trace' | 'evidence' | 'tools' | 'fallback' | 'metrics' | 'upload' | 'alerts'>('trace')
  const [inputValue, setInputValue] = useState('')
  const sessionId = useRef(`s-demo-${genId().slice(0, 8)}`)
  const userId = 'u001'
  const scrollRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    }, 100)
  }, [])

  async function handleSend(query: string) {
    setError('')
    const userMsg: ChatMessage = { id: genId(), role: 'user', content: query, timestamp: Date.now() }
    setMessages(prev => [...prev, userMsg])
    scrollToBottom()
    setLoading(true)
    try {
      const response = await sendChatMessage({ query, user_id: userId, session_id: sessionId.current })
      const aiMsg: ChatMessage = { id: genId(), role: 'assistant', content: response.answer, timestamp: Date.now(), response }
      setMessages(prev => [...prev, aiMsg])
      setActiveResponse(response)
    } catch (e) {
      setError(e instanceof Error ? e.message : '请求失败')
    } finally {
      setLoading(false)
      scrollToBottom()
    }
  }

  function handleDemoClick(q: string) {
    setInputValue(q)
  }

  const hasEvidence = (activeResponse?.citations?.length ?? 0) > 0
  const hasTools = (activeResponse?.tool_results?.length ?? 0) > 0
  const hasFallback = !!activeResponse?.fallback_reason || activeResponse?.need_human

  return (
    <div className="flex h-screen bg-gray-50">
      {/* ========== LEFT SIDEBAR ========== */}
      <aside className="w-[240px] shrink-0 bg-gray-900 text-gray-100 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center">
              <MessageSquare size={16} className="text-white" />
            </div>
            <div>
              <div className="font-bold text-xs">Enterprise RAG</div>
              <div className="text-[10px] text-gray-400">QA Console</div>
            </div>
          </div>
        </div>

        {/* Session info */}
        <div className="p-3 border-b border-gray-700">
          <div className="text-[10px] text-gray-500 font-medium uppercase tracking-wider mb-1">会话</div>
          <div className="text-[11px] font-mono text-gray-300 truncate">{sessionId.current}</div>
          <div className="text-[10px] text-gray-500">用户: {userId} | {messages.length} 条消息</div>
        </div>

        {/* Demo questions */}
        <div className="flex-1 overflow-y-auto p-3">
          <div className="text-[10px] text-gray-500 font-medium uppercase tracking-wider mb-2">Demo 问题</div>
          <div className="space-y-1">
            {DEMO_QUESTIONS.map((dq, i) => (
              <button
                key={i}
                onClick={() => handleDemoClick(dq.q)}
                className="w-full text-left rounded-lg px-2.5 py-2 hover:bg-gray-800 transition-colors group"
              >
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-400 shrink-0">{dq.label}</span>
                </div>
                <div className="text-[11px] text-gray-300 mt-0.5 group-hover:text-white line-clamp-2 leading-relaxed">{dq.q}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="p-3 border-t border-gray-700">
          <SystemStatusBadge />
        </div>
      </aside>

      {/* ========== CENTER CHAT ========== */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header with capability tags */}
        <header className="shrink-0 bg-white border-b border-gray-200 px-5 py-3">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="font-bold text-gray-800">Enterprise Agentic RAG QA Console</h1>
              <p className="text-[11px] text-gray-500">LangGraph Multi-Agent · 企业多智能体问答系统</p>
            </div>
            <MetricsMiniBar />
            {error && <div className="text-xs text-red-500 bg-red-50 rounded px-2 py-1 ml-2">{error}</div>}
          </div>
          {/* Capability tags */}
          <div className="flex flex-wrap gap-1.5 mt-2">
            {CAPABILITY_TAGS.map(tag => (
              <span key={tag} className="inline-flex items-center rounded-full bg-blue-50 px-2.5 py-0.5 text-[10px] font-medium text-blue-700 border border-blue-100">
                {tag}
              </span>
            ))}
          </div>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center space-y-3 max-w-md">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-blue-50 flex items-center justify-center">
                  <MessageSquare size={32} className="text-blue-400" />
                </div>
                <p className="font-medium text-gray-600">体验企业多智能体问答</p>
                <p className="text-sm">左侧 Demo 问题点击即可提问，或直接输入您的问题</p>
              </div>
            </div>
          )}
          {messages.map(msg => (
            <ChatMessageBubble key={msg.id} msg={msg} sessionId={sessionId.current} />
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-sm">🤖</div>
              <div className="rounded-xl bg-white border px-4 py-3 shadow-sm">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" />
                  <span className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" style={{ animationDelay: '0.2s' }} />
                  <span className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" style={{ animationDelay: '0.4s' }} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <ChatInput onSend={handleSend} disabled={loading} externalValue={inputValue} onValueConsumed={() => setInputValue('')} />
      </main>

      {/* ========== RIGHT PANEL ========== */}
      <aside className="w-[340px] shrink-0 bg-white border-l border-gray-200 flex flex-col">
        <div className="flex border-b border-gray-200 overflow-x-auto">
          <TabBtn active={rightTab === 'trace'} onClick={() => setRightTab('trace')} icon={<Activity size={13} />} label="追踪" />
          <TabBtn active={rightTab === 'evidence'} onClick={() => setRightTab('evidence')} icon={<Database size={13} />} label="证据" dot={hasEvidence} />
          <TabBtn active={rightTab === 'tools'} onClick={() => setRightTab('tools')} icon={<Wrench size={13} />} label="工具" dot={hasTools} />
          <TabBtn active={rightTab === 'fallback'} onClick={() => setRightTab('fallback')} icon={<ShieldAlert size={13} />} label="兜底" dot={hasFallback} />
          <TabBtn active={rightTab === 'metrics'} onClick={() => setRightTab('metrics')} icon={<BarChart3 size={13} />} label="指标" />
          <TabBtn active={rightTab === 'upload'} onClick={() => setRightTab('upload')} icon={<Upload size={13} />} label="上传" />
          <TabBtn active={rightTab === 'alerts'} onClick={() => setRightTab('alerts')} icon={<ShieldAlert size={13} />} label="告警" />
        </div>
        <div className="flex-1 overflow-y-auto">
          {rightTab === 'trace' && <AgentTracePanel response={activeResponse} />}
          {rightTab === 'evidence' && <RAGEvidencePanel response={activeResponse} />}
          {rightTab === 'tools' && <ToolCallsPanel response={activeResponse} />}
          {rightTab === 'fallback' && <FallbackPanel response={activeResponse} />}
          {rightTab === 'metrics' && <FullMetricsPanel />}
          {rightTab === 'upload' && <FileUpload />}
          {rightTab === 'alerts' && <BadCasePanel />}
        </div>
      </aside>
    </div>
  )
}

function TabBtn({ active, onClick, icon, label, dot }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string; dot?: boolean }) {
  return (
    <button
      onClick={onClick}
      className={`relative flex-1 py-2.5 text-[11px] font-medium flex items-center justify-center gap-1 ${
        active ? 'text-blue-600 border-b-2 border-blue-500 bg-blue-50/50' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
      }`}
    >
      {icon}
      <span className="hidden lg:inline">{label}</span>
      {dot && <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-orange-400" />}
    </button>
  )
}

function FullMetricsPanel() {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  const [ml, setMl] = useState(false)

  useEffect(() => {
    async function load() {
      setMl(true)
      try {
        const res = await fetch('/metrics')
        const data = await res.json()
        setMetrics(data.metrics)
      } catch { /* ignore */ }
      finally { setMl(false) }
    }
    load()
  }, [])

  if (!metrics) return <div className="p-4 text-xs text-gray-400 text-center">加载中...</div>

  const m = metrics as Record<string, number | Record<string, number>>
  return (
    <div className="p-4 space-y-2 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <MiniCard label="总请求" value={String(m.total_requests ?? 0)} />
        <MiniCard label="成功率" value={`${((m.success_rate as number ?? 0) * 100).toFixed(1)}%`} />
        <MiniCard label="平均延迟" value={`${(m.avg_latency_ms as number ?? 0).toFixed(1)}ms`} />
        <MiniCard label="检索命中" value={`${((m.retrieval_hit_rate as number ?? 0) * 100).toFixed(1)}%`} />
        <MiniCard label="校验通过" value={`${((m.verification_pass_rate as number ?? 0) * 100).toFixed(1)}%`} color="green" />
        <MiniCard label="工具成功" value={`${((m.tool_success_rate as number ?? 0) * 100).toFixed(1)}%`} color="green" />
        <MiniCard label="兜底率" value={`${((m.fallback_rate as number ?? 0) * 100).toFixed(1)}%`} color="amber" />
        <MiniCard label="人工升级" value={`${((m.human_fallback_rate as number ?? 0) * 100).toFixed(1)}%`} color="red" />
      </div>
      {m.intent_distribution && (
        <div>
          <div className="text-xs font-semibold text-gray-600 mt-2 mb-1">意图分布</div>
          {Object.entries(m.intent_distribution as Record<string, number>).sort(([,a],[,b]) => b - a).map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs py-0.5">
              <span className="text-gray-600">{k}</span>
              <span className="font-mono">{v}</span>
            </div>
          ))}
        </div>
      )}
      <button onClick={() => {
        setMl(true)
        fetch('/metrics').then(r => r.json()).then(d => setMetrics(d.metrics)).finally(() => setMl(false))
      }} disabled={ml} className="text-[10px] text-blue-500 hover:text-blue-700 mt-2">刷新</button>
    </div>
  )
}

function MiniCard({ label, value, color }: { label: string; value: string; color?: string }) {
  const colors: Record<string, string> = {
    green: 'bg-green-50 border-green-200',
    amber: 'bg-amber-50 border-amber-200',
    red: 'bg-red-50 border-red-200',
    default: 'bg-gray-50 border-gray-200',
  }
  return (
    <div className={`rounded-lg border px-2.5 py-1.5 ${colors[color ?? 'default']}`}>
      <div className="text-[10px] text-gray-500">{label}</div>
      <div className="font-mono font-semibold text-sm">{value}</div>
    </div>
  )
}
