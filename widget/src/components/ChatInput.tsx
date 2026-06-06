import { useState, useRef, useEffect } from 'react'
import { SendIcon, BrainIcon } from './icons'

interface Props {
  onSend: (query: string, deepThinking: boolean) => void
  disabled: boolean
}

/**
 * 聊天输入框 — contentEditable + 深度思考开关 + 发送按钮
 * 模仿华为智能客服的输入区域设计
 */
export default function ChatInput({ onSend, disabled }: Props) {
  const [deepThinking, setDeepThinking] = useState(true)
  const [hasContent, setHasContent] = useState(false)
  const inputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!disabled) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [disabled])

  function handleInput() {
    const text = inputRef.current?.innerText ?? ''
    setHasContent(text.trim().length > 0)
  }

  function handleSend() {
    const text = inputRef.current?.innerText?.trim()
    if (!text || disabled) return
    onSend(text, deepThinking)
    // 清空
    if (inputRef.current) {
      inputRef.current.innerHTML = ''
      inputRef.current.innerText = ''
      setHasContent(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="shrink-0 border-t border-gray-200 bg-white">
      {/* 输入区域 */}
      <div className="px-4 pt-3">
        <div className="flex items-start gap-2 rounded-xl border border-gray-300 bg-gray-50 px-3 py-2 focus-within:border-blue-400 focus-within:bg-white focus-within:ring-2 focus-within:ring-blue-100 transition-all">
          {/* ContentEditable 输入框 */}
          <div
            ref={inputRef}
            contentEditable={!disabled}
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            data-placeholder="请告诉我您遇到的问题"
            className="flex-1 min-h-[24px] max-h-[120px] overflow-y-auto text-sm outline-none empty:before:content-[attr(data-placeholder)] empty:before:text-gray-400 py-1"
            suppressContentEditableWarning
          />

          {/* 发送按钮 */}
          <button
            onClick={handleSend}
            disabled={disabled || !hasContent}
            className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all ${
              hasContent && !disabled
                ? 'bg-blue-500 text-white hover:bg-blue-600 shadow-sm'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
          >
            <SendIcon size={15} />
          </button>
        </div>
      </div>

      {/* 底部操作栏 */}
      <div className="flex items-center justify-between px-4 py-2">
        {/* 深度思考开关 */}
        <button
          onClick={() => setDeepThinking(!deepThinking)}
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium transition-all ${
            deepThinking
              ? 'bg-amber-50 text-amber-700 border border-amber-200'
              : 'bg-gray-100 text-gray-500 border border-gray-200'
          }`}
        >
          <BrainIcon size={12} className={deepThinking ? 'text-amber-500' : 'text-gray-400'} />
          深度思考
          {deepThinking && (
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
          )}
        </button>

        <p className="text-[10px] text-gray-300">
          Enter 发送 · Shift+Enter 换行
        </p>
      </div>
    </div>
  )
}
