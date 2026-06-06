import type { Citation, StreamEvent } from '../types/chat'

/**
 * SSE 流式聊天 — 实时推送回答和深度思考内容
 */
export async function* streamChat(
  query: string,
  userId: string,
  sessionId: string,
  enableDeepThinking: boolean,
): AsyncGenerator<StreamEvent> {
  const resp = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      user_id: userId,
      session_id: sessionId,
      deep_thinking: enableDeepThinking,
    }),
  })

  if (!resp.ok) {
    const err = await resp.text().catch(() => 'Unknown error')
    throw new Error(`请求失败 (${resp.status}): ${err}`)
  }

  const reader = resp.body?.getReader()
  if (!reader) throw new Error('浏览器不支持流式读取')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data: ')) continue
      const json = trimmed.slice(6)
      try {
        const event = JSON.parse(json) as StreamEvent
        yield event
      } catch {
        // skip unparseable lines
      }
    }
  }
}

/**
 * 获取推荐问题列表
 */
export async function getSuggestions(): Promise<{ suggestions: { id: string; label: string; question: string; icon?: string }[] }> {
  const resp = await fetch('/api/suggestions')
  if (!resp.ok) throw new Error('获取推荐问题失败')
  return resp.json()
}

/**
 * 提交反馈
 */
export async function sendFeedback(
  traceId: string,
  sessionId: string,
  thumbsUp: boolean,
  feedbackText?: string,
  userId?: string,
): Promise<{ received: boolean }> {
  const resp = await fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      trace_id: traceId,
      session_id: sessionId,
      thumbs_up: thumbsUp,
      feedback_text: feedbackText ?? '',
      user_id: userId ?? 'anonymous',
    }),
  })
  if (!resp.ok) throw new Error('反馈提交失败')
  return resp.json()
}
