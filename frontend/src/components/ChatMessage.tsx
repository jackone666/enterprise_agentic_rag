import type { ChatMessage as ChatMessageType } from '../types/chat'
import { CheckCircle, XCircle, AlertTriangle, Tag, Hash, FileText } from './icons'
import FeedbackButtons from './FeedbackButtons'

export default function ChatMessage({ msg, sessionId }: { msg: ChatMessageType; sessionId: string }) {
  const isUser = msg.role === 'user'
  const r = msg.response

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Avatar */}
      <div className={`shrink-0 w-9 h-9 rounded-xl flex items-center justify-center text-sm shadow-sm ${isUser ? 'order-2 bg-blue-500 text-white' : 'bg-gradient-to-br from-gray-100 to-gray-200 text-gray-600'}`}>
        {isUser ? '👤' : '🤖'}
      </div>

      {/* Content */}
      <div className={`max-w-[75%] ${isUser ? 'order-1' : ''}`}>
        {/* User message */}
        {isUser && (
          <div className="rounded-2xl rounded-tr-sm bg-blue-500 text-white px-4 py-2.5 text-sm shadow-sm">
            {msg.content}
          </div>
        )}

        {/* AI response */}
        {!isUser && r && (
          <div className="space-y-2">
            {/* Answer card */}
            <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 shadow-sm overflow-hidden">
              {/* Status header bar */}
              <div className="flex items-center gap-2 px-4 py-2 bg-gray-50 border-b border-gray-100">
                {/* Intent badge */}
                <span className="inline-flex items-center gap-1 rounded-full bg-purple-50 px-2 py-0.5 text-[10px] font-medium text-purple-700 border border-purple-100">
                  <Tag size={10} /> {r.intent ?? 'unknown'}
                </span>

                {/* Verification badge */}
                {r.verified !== undefined && (
                  <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium border ${
                    r.verified
                      ? 'bg-green-50 text-green-700 border-green-100'
                      : 'bg-red-50 text-red-700 border-red-100'
                  }`}>
                    {r.verified ? <CheckCircle size={10} /> : <XCircle size={10} />}
                    {r.verified ? '已通过答案校验' : '答案未通过校验'}
                  </span>
                )}

                {/* Human fallback */}
                {r.need_human && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700 border border-amber-100">
                    <AlertTriangle size={10} /> 人工兜底
                  </span>
                )}

                {/* Trace */}
                <span className="text-[10px] text-gray-400 font-mono ml-auto">
                  <Hash size={9} className="inline mr-0.5" />{r.trace_id?.slice(0, 8) ?? '-'}
                </span>
              </div>

              {/* Answer body */}
              <div className="px-4 py-3">
                <div className="text-sm leading-relaxed whitespace-pre-wrap">{r.answer}</div>
              </div>

              {/* Citations footer */}
              {r.citations && r.citations.length > 0 && (
                <div className="px-4 py-2.5 bg-gray-50/50 border-t border-gray-100">
                  <div className="flex flex-wrap gap-1.5">
                    {r.citations.map((c, i) => (
                      <span key={i} className="inline-flex items-center gap-1 rounded-lg bg-white border border-gray-200 px-2 py-1 text-[10px] shadow-sm">
                        <FileText size={10} className="text-blue-400" />
                        <span className="font-medium">[{c.index ?? i + 1}]</span>
                        <span className="text-gray-600">{c.source ?? 'unknown'}</span>
                        <span className="text-blue-500 font-mono">{(c.relevance_score ?? c.score ?? 0).toFixed(2)}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Tool results summary */}
              {r.tool_results && r.tool_results.length > 0 && (
                <div className="px-4 py-2 border-t border-gray-100">
                  <div className="flex flex-wrap gap-1.5">
                    {r.tool_results.map((tr, i) => (
                      <span key={i} className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium ${
                        tr.success
                          ? 'bg-green-50 text-green-700'
                          : 'bg-red-50 text-red-700'
                      }`}>
                        {tr.success ? '✅' : '❌'} {tr.tool_name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Feedback */}
            <div className="px-1">
              <FeedbackButtons traceId={r.trace_id ?? ''} sessionId={sessionId} />
            </div>
          </div>
        )}

        {/* Loading placeholder */}
        {!isUser && !r && (
          <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 px-4 py-3 text-sm shadow-sm text-gray-400">
            <div className="flex gap-1.5">
              <span className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" />
              <span className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" style={{ animationDelay: '0.2s' }} />
              <span className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" style={{ animationDelay: '0.4s' }} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
