/** 推荐问题 */
export interface Suggestion {
  id: string
  label: string
  question: string
  icon?: string
}

/** 聊天消息 */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  /** 深度思考内容 */
  thinking?: string
  /** 回答是否完成 */
  complete?: boolean
  /** 引用来源 */
  citations?: Citation[]
  /** 反馈状态 */
  feedback?: 'up' | 'down' | null
}

/** 引用 */
export interface Citation {
  index?: number
  source: string
  chunk_id?: string
  score?: number
  relevance_score?: number
}

/** SSE 流事件 */
export interface StreamEvent {
  type: 'start' | 'node_end' | 'thinking' | 'answer_chunk' | 'done' | 'error' | 'end'
  trace_id?: string
  query?: string
  node?: string
  data?: Record<string, unknown>
  answer?: string
  citations?: Citation[]
  verified?: boolean
  intent?: string
  message?: string
  content?: string
  thinking_content?: string
}
