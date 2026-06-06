import { useState, useRef, useCallback, useEffect } from 'react'
import type { ChatMessage, Suggestion } from '../types/chat'
import { streamChat, getSuggestions } from '../api/chat'
import ChatMessageBubble from './ChatMessage'
import ChatInput from './ChatInput'
import WelcomeScreen from './WelcomeScreen'

function genId() {
  return Math.random().toString(36).slice(2, 10)
}

/** 默认推荐问题（后备）- 请求失败时使用 */
const DEFAULT_SUGGESTIONS: Suggestion[] = [
  { id: 'develop', label: '开发入门', question: '鸿蒙应用开发如何入门？', icon: '💻' },
  { id: 'upgrade', label: '系统升级', question: '如何升级HarmonyOS 6？', icon: '⬆️' },
  { id: 'api', label: 'API 使用', question: 'HarmonyOS 网络请求 API 怎么用？', icon: '🔌' },
  { id: 'error', label: '错误排查', question: '应用闪退怎么排查？', icon: '🔧' },
  { id: 'distribute', label: '应用分发', question: '如何发布应用到华为应用市场？', icon: '📦' },
]

/**
 * 主聊天控件 — 完整智能客服界面
 *
 * 支持两种模式：
 * - page: 独立全页（路由 /chat-page）
 * - embedded: 嵌入侧边面板（由 FloatingWidget 控制）
 */
export default function ChatWidget({ mode = 'page' }: { mode?: 'page' | 'embedded' }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<Suggestion[]>(DEFAULT_SUGGESTIONS)
  const [error, setError] = useState('')
  const sessionId = useRef(`s-${genId().slice(0, 8)}`)
  const userId = useRef(`u-${genId().slice(0, 6)}`)
  const scrollRef = useRef<HTMLDivElement>(null)
  const traceIdRef = useRef('')

  // 加载推荐问题
  useEffect(() => {
    getSuggestions()
      .then((r) => {
        if (r.suggestions?.length) setSuggestions(r.suggestions)
      })
      .catch(() => {
        // 使用默认推荐问题
      })
  }, [])

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      })
    }, 100)
  }, [])

  async function handleSend(query: string, deepThinking: boolean) {
    setError('')
    setLoading(true)
    traceIdRef.current = ''

    const userMsg: ChatMessage = {
      id: genId(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    const assistantMsg: ChatMessage = {
      id: genId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      thinking: '',
      complete: false,
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    scrollToBottom()

    try {
      let answerText = ''
      let thinkingText = ''
      let citationsData: ChatMessage['citations'] = []

      for await (const event of streamChat(
        query,
        userId.current,
        sessionId.current,
        deepThinking,
      )) {
        switch (event.type) {
          case 'start':
            traceIdRef.current = event.trace_id ?? ''
            break

          case 'thinking':
            thinkingText += event.thinking_content ?? ''
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, thinking: thinkingText }
                  : m,
              ),
            )
            break

          case 'answer_chunk':
            answerText += event.content ?? ''
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, content: answerText }
                  : m,
              ),
            )
            break

          case 'node_end':
            // 某些后端节点完成事件可能携带中间产出
            break

          case 'done':
            answerText = event.answer ?? answerText
            citationsData = (event.citations as ChatMessage['citations']) ?? []
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? {
                      ...m,
                      content: answerText || m.content,
                      thinking: thinkingText || m.thinking,
                      complete: true,
                      citations: citationsData,
                    }
                  : m,
              ),
            )
            break

          case 'error':
            setError(event.message ?? '处理出错')
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, content: `抱歉，${event.message ?? '处理您的请求时出现了错误，请稍后再试。'}`, complete: true }
                  : m,
              ),
            )
            break
        }
      }

      // 如果流式结束没有 done 事件，做收尾处理
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id && !m.complete
            ? { ...m, content: m.content || '抱歉，回答生成中断，请重试。', complete: true }
            : m,
        ),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : '请求失败')
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? { ...m, content: '抱歉，连接出现了问题，请稍后再试。', complete: true }
            : m,
        ),
      )
    } finally {
      setLoading(false)
      scrollToBottom()
    }
  }

  const showWelcome = messages.length === 0 && !loading

  return (
    <div
      className={`flex flex-col bg-gray-50 ${
        mode === 'embedded' ? 'h-full' : 'h-screen'
      }`}
    >
      {/* 顶部导航栏 */}
      <header className="shrink-0 bg-primary border-b border-gray-800 px-5 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
              <span className="text-white font-bold text-xs">AI</span>
            </div>
            <div>
              <div className="text-sm font-semibold text-white">智能客服</div>
              <div className="text-[10px] text-gray-400">Powered by Enterprise Agentic RAG</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {error && (
              <span className="text-[10px] text-red-400 bg-red-900/30 rounded px-2 py-0.5">
                {error}
              </span>
            )}
            <span className="inline-flex items-center gap-1 rounded-full bg-green-900/30 px-2 py-0.5 text-[10px] text-green-400">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
              在线
            </span>
          </div>
        </div>
      </header>

      {/* 对话区域 */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {showWelcome ? (
          <WelcomeScreen
            suggestions={suggestions}
            onQuestionClick={(q) => handleSend(q, true)}
          />
        ) : (
          <div className="max-w-[800px] mx-auto p-4 space-y-4">
            {messages.map((msg) => (
              <ChatMessageBubble
                key={msg.id}
                msg={msg}
                sessionId={sessionId.current}
              />
            ))}
          </div>
        )}
      </div>

      {/* 输入区域 */}
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  )
}
