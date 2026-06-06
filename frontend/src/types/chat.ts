// === Request types ===
export interface ChatRequest {
  query: string
  user_id: string
  session_id: string
}

export interface FeedbackRequest {
  trace_id: string
  session_id: string
  thumbs_up: boolean
  feedback_text: string
  user_id: string
}

// === Response types ===
export interface Citation {
  index?: number
  source: string
  chunk_id?: string
  score?: number
  relevance_score?: number
}

export interface ToolResult {
  tool_name: string
  success: boolean
  output: unknown
  error: string | null
  latency_ms: number
}

export interface NodeEvent {
  node_name: string
  event_type: string
  latency_ms: number
  success: boolean
  error: string
  timestamp: number
  timestamp_iso?: string
  input_summary?: string
  output_summary?: string
}

export interface RetrievedDoc {
  source: string
  content: string
  score: number
  chunk_id?: string
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  intent: string
  need_human: boolean
  trace_id: string
  verification_reason: string
  verified?: boolean
  tool_results: ToolResult[]
  tool_errors: string[]
  retrieved_docs: RetrievedDoc[]
  fallback_reason: string
  recovery_action: string
  recoverable: boolean
  retry_count: Record<string, number>
  retry_history: Record<string, unknown>[]
  node_events_count: number
  node_events: NodeEvent[]
  retrieval_events: Record<string, unknown>[]
  verification_events: Record<string, unknown>[]
  chat_history_count: number
  session_summary: string
  metrics_snapshot: Record<string, unknown>
  auto_captured?: boolean
  eval_result?: {
    overall: number
    precision: number
    recall: number
    faithfulness: number
    relevance: number
    passing: boolean
  }
  pipeline_trace?: {
    total_latency_ms: number
    node_count: number
    steps: { name: string; latency_ms: number; success: boolean; error: string }[]
    backend: string
  }
  retrieval_backend?: string
}

// === Metrics types ===
export interface MetricsSnapshot {
  total_requests: number
  total_success: number
  success_rate: number
  avg_latency_ms: number
  intent_distribution: Record<string, number>
  retrieval_hit_rate: number
  verification_pass_rate: number
  tool_success_rate: number
  fallback_rate: number
  human_fallback_rate: number
  uptime_seconds: number
}

export interface MetricsResponse {
  metrics: MetricsSnapshot
}

// === Feedback response ===
export interface FeedbackResponse {
  received: boolean
  auto_captured: boolean
  reason: string
}

// === Chat message for UI ===
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  response?: ChatResponse
}
