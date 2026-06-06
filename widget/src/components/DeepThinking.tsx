import { useState } from 'react'
import { BrainIcon, ChevronDownIcon, ChevronUpIcon } from './icons'

interface Props {
  content: string
  complete?: boolean
}

/**
 * 深度思考（CoT）面板 — 可折叠展示 AI 推理过程
 * 模仿华为智能客服的「深度思考」功能
 */
export default function DeepThinking({ content, complete }: Props) {
  const [expanded, setExpanded] = useState(true)

  if (!content) return null

  return (
    <div className="border border-amber-200 rounded-lg overflow-hidden bg-amber-50/30">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-amber-50/50 transition-colors"
      >
        <BrainIcon size={14} className="text-amber-500" />
        <span className="text-xs font-medium text-amber-700">
          {complete ? '思考完成' : '正在思考...'}
        </span>
        {complete && (
          <span className="text-[10px] text-amber-400 ml-1">点击{expanded ? '收起' : '展开'}</span>
        )}
        <span className="ml-auto text-amber-400">
          {expanded ? <ChevronUpIcon size={14} /> : <ChevronDownIcon size={14} />}
        </span>
      </button>
      {expanded && (
        <div className="thinking-expand px-3 pb-3 border-t border-amber-100">
          <pre className="text-[11px] leading-relaxed text-amber-800 whitespace-pre-wrap font-sans mt-2 max-h-[300px] overflow-y-auto">
            {content}
          </pre>
        </div>
      )}
    </div>
  )
}
