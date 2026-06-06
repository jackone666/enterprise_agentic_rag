import { useState, useRef, useEffect } from 'react'
import { Send, Sparkles } from './icons'

interface Props {
  onSend: (query: string) => void
  disabled: boolean
  externalValue?: string
  onValueConsumed?: () => void
}

export default function ChatInput({ onSend, disabled, externalValue, onValueConsumed }: Props) {
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  // Consume external value (from demo question clicks)
  useEffect(() => {
    if (externalValue && externalValue.trim()) {
      setQuery(externalValue)
      onValueConsumed?.()
      // Auto-focus
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [externalValue, onValueConsumed])

  useEffect(() => {
    if (!disabled && inputRef.current) inputRef.current.focus()
  }, [disabled])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    onSend(query.trim())
    setQuery('')
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-3 border-t border-gray-200 bg-white">
      <div className="flex-1 relative">
        <input
          ref={inputRef}
          type="text"
          className="w-full rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent disabled:opacity-50 pr-8"
          placeholder="输入您的问题，例如：如何重置密码？"
          value={query}
          onChange={e => setQuery(e.target.value)}
          disabled={disabled}
        />
        {!query && !disabled && (
          <Sparkles size={14} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-300" />
        )}
      </div>
      <button
        type="submit"
        disabled={disabled || !query.trim()}
        className="rounded-lg bg-blue-500 px-4 py-2.5 text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 transition-colors"
      >
        <Send size={16} />
        <span className="hidden sm:inline font-medium">发送</span>
      </button>
    </form>
  )
}
