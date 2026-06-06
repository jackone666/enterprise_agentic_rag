import type { ChatResponse } from '../types/chat'
import { Database, FileText } from './icons'

export default function RAGEvidencePanel({ response }: { response: ChatResponse | null }) {
  if (!response) {
    return (
      <div className="p-4 text-sm text-gray-400 text-center">
        <Database size={24} className="mx-auto mb-2 opacity-50" />
        <p>等待请求...</p>
      </div>
    )
  }

  const citations = response.citations ?? []

  return (
    <div className="p-4 space-y-3 text-sm">
      <div className="flex items-center gap-1.5 font-semibold text-gray-700">
        <Database size={14} /> RAG 检索证据
      </div>

      {citations.length === 0 ? (
        <div className="text-xs text-gray-400">无检索结果（知识库未命中或走兜底流程）</div>
      ) : (
        <div className="space-y-2">
          {citations.map((c, i) => (
            <div key={i} className="rounded-lg border border-gray-200 p-3 bg-gray-50">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-xs font-medium text-gray-800">
                  [{c.index ?? i + 1}] {c.source ?? 'unknown'}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
                  {(c.relevance_score ?? c.score ?? 0).toFixed(2)}
                </span>
              </div>
              {c.chunk_id && (
                <div className="text-[10px] text-gray-400 font-mono">{c.chunk_id}</div>
              )}
              <div className="flex items-center gap-1 text-[10px] text-gray-500 mt-1">
                <FileText size={10} />
                <span>来源文档</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Retrieval events */}
      {response.retrieval_events && (response.retrieval_events as unknown[]).length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="text-xs font-semibold text-gray-600 mb-1">检索事件 ({(response.retrieval_events as unknown[]).length})</div>
        </div>
      )}
    </div>
  )
}
