import { useState } from 'react'
import { ThumbsUp, ThumbsDown, Send } from './icons'
import { sendFeedback } from '../api/feedback'

interface Props {
  traceId: string
  sessionId: string
  userId?: string
}

export default function FeedbackButtons({ traceId, sessionId, userId = 'anonymous' }: Props) {
  const [sent, setSent] = useState<'up' | 'down' | null>(null)
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)

  async function handle(thumbsUp: boolean) {
    setSending(true)
    try {
      await sendFeedback({
        trace_id: traceId,
        session_id: sessionId,
        thumbs_up: thumbsUp,
        feedback_text: text,
        user_id: userId,
      })
      setSent(thumbsUp ? 'up' : 'down')
    } catch {
      // still mark as sent to prevent double-click
      setSent(thumbsUp ? 'up' : 'down')
    } finally {
      setSending(false)
    }
  }

  if (sent) {
    return <div className="text-xs text-gray-500 mt-1">{sent === 'up' ? '👍 感谢反馈!' : '👎 已记录，我们会改进'}</div>
  }

  return (
    <div className="mt-2 space-y-1.5">
      <div className="flex gap-1">
        <button
          onClick={() => handle(true)}
          disabled={sending}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs border border-gray-200 hover:bg-green-50 hover:border-green-300 disabled:opacity-50"
        >
          <ThumbsUp size={12} /> 有帮助
        </button>
        <button
          onClick={() => handle(false)}
          disabled={sending}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs border border-gray-200 hover:bg-red-50 hover:border-red-300 disabled:opacity-50"
        >
          <ThumbsDown size={12} /> 没帮助
        </button>
      </div>
      <div className="flex gap-1">
        <input
          className="flex-1 rounded border border-gray-200 px-2 py-1 text-xs focus:outline-none focus:border-blue-400"
          placeholder="补充说明（可选）"
          value={text}
          onChange={e => setText(e.target.value)}
        />
        <button
          onClick={() => handle(false)}
          disabled={sending || !text}
          className="rounded bg-gray-100 px-2 py-1 hover:bg-gray-200 disabled:opacity-50"
          title="提交反馈"
        >
          <Send size={12} />
        </button>
      </div>
    </div>
  )
}
