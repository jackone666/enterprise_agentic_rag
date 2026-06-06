import { useState } from 'react'
import { ThumbsUpIcon, ThumbsDownIcon, CopyIcon, CheckIcon, RefreshCwIcon } from './icons'
import { sendFeedback } from '../api/chat'

interface Props {
  traceId: string
  sessionId: string
  content: string
  userId?: string
}

/**
 * 消息操作按钮 — 复制、点赞、点踩
 */
export default function FeedbackButtons({ traceId, sessionId, content, userId }: Props) {
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)
  const [copied, setCopied] = useState(false)
  const [sending, setSending] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // fallback
      const ta = document.createElement('textarea')
      ta.value = content
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  async function handleFeedback(type: 'up' | 'down') {
    if (sending || feedback) return
    setSending(true)
    try {
      await sendFeedback(traceId, sessionId, type === 'up', undefined, userId)
      setFeedback(type)
    } catch {
      // still show feedback state to prevent spam
      setFeedback(type)
    } finally {
      setSending(false)
    }
  }

  async function handleRetry() {
    setFeedback(null)
  }

  return (
    <div className="flex items-center gap-1">
      {/* 复制 */}
      <button
        onClick={handleCopy}
        className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
        title="复制回答"
      >
        {copied ? <CheckIcon size={13} className="text-green-500" /> : <CopyIcon size={13} />}
        {copied ? '已复制' : '复制'}
      </button>

      {/* 点赞 */}
      <button
        onClick={() => handleFeedback('up')}
        disabled={sending}
        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors ${
          feedback === 'up'
            ? 'text-green-600 bg-green-50'
            : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
        }`}
        title="有帮助"
      >
        <ThumbsUpIcon size={13} />
        有帮助
      </button>

      {/* 点踩 */}
      <button
        onClick={() => handleFeedback('down')}
        disabled={sending}
        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors ${
          feedback === 'down'
            ? 'text-red-600 bg-red-50'
            : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
        }`}
        title="没帮助"
      >
        <ThumbsDownIcon size={13} />
        没帮助
      </button>

      {/* 重试 */}
      {feedback && (
        <button
          onClick={handleRetry}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          title="重新评价"
        >
          <RefreshCwIcon size={12} />
        </button>
      )}
    </div>
  )
}
