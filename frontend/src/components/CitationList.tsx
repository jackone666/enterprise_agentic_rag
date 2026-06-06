import type { Citation } from '../types/chat'

export default function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations || citations.length === 0) return null

  return (
    <div className="mt-2 text-xs">
      <div className="font-semibold text-gray-600 mb-1">📚 参考来源</div>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((c, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2 py-0.5 text-blue-700 border border-blue-200"
          >
            <span className="font-mono text-[10px]">[{c.index ?? i + 1}]</span>
            <span>{c.source ?? 'unknown'}</span>
            {c.relevance_score != null && (
              <span className="text-blue-400">{(c.relevance_score as number).toFixed(2)}</span>
            )}
            {c.score != null && c.relevance_score == null && (
              <span className="text-blue-400">{(c.score as number).toFixed(2)}</span>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}
