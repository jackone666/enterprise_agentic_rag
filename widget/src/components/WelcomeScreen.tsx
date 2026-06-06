import { SparklesIcon } from './icons'

interface Suggestion {
  id: string
  label: string
  question: string
  icon?: string
}

interface Props {
  suggestions: Suggestion[]
  onQuestionClick: (question: string) => void
}

/**
 * 欢迎页 — 模仿华为智能客服的欢迎区域
 * 包含问候语 + 推荐问题列表
 */
export default function WelcomeScreen({ suggestions, onQuestionClick }: Props) {
  // 默认图标映射
  const defaultIcons: Record<string, string> = {
    login: '🔑',
    register: '📝',
    event: '🎉',
    upgrade: '⬆️',
    incentive: '💰',
    develop: '💻',
    document: '📄',
    question: '❓',
    service: '🎧',
    default: '💡',
  }

  function getIcon(suggestion: Suggestion): string {
    if (suggestion.icon) return suggestion.icon
    for (const [key, icon] of Object.entries(defaultIcons)) {
      if (suggestion.id.includes(key) || suggestion.label.includes(key)) return icon
    }
    return defaultIcons.default
  }

  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="w-full max-w-[720px] text-center">
        {/* 问候语 */}
        <div className="mb-2">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-50 mb-4">
            <SparklesIcon size={28} className="text-blue-400" />
          </div>
          <h1 className="text-xl font-semibold text-gray-800 mb-1">
            嗨，我是您的智能客服
          </h1>
          <p className="text-sm text-gray-400">
            请问有什么可以帮您？
          </p>
        </div>

        {/* 推荐问题 */}
        <div className="mt-6">
          {/* 第一行：2 列 */}
          <div className="grid grid-cols-2 gap-2.5 mb-2.5">
            {suggestions.slice(0, 2).map((s) => (
              <button
                key={s.id}
                onClick={() => onQuestionClick(s.question)}
                className="flex items-center gap-2.5 text-left rounded-xl border border-gray-200 bg-white px-4 py-3 hover:border-blue-300 hover:shadow-sm transition-all group"
              >
                <span className="text-lg shrink-0">{getIcon(s)}</span>
                <span className="text-sm text-gray-700 group-hover:text-blue-600">
                  {s.question}
                </span>
              </button>
            ))}
          </div>

          {/* 第二行：3 列 */}
          <div className="grid grid-cols-3 gap-2.5">
            {suggestions.slice(2, 5).map((s) => (
              <button
                key={s.id}
                onClick={() => onQuestionClick(s.question)}
                className="flex items-center gap-2 text-left rounded-xl border border-gray-200 bg-white px-3.5 py-3 hover:border-blue-300 hover:shadow-sm transition-all group"
              >
                <span className="text-base shrink-0">{getIcon(s)}</span>
                <span className="text-sm text-gray-700 group-hover:text-blue-600 leading-snug">
                  {s.question}
                </span>
              </button>
            ))}
          </div>

          {/* 第三行：剩余推荐 */}
          {suggestions.length > 5 && (
            <div className="grid grid-cols-3 gap-2.5 mt-2.5">
              {suggestions.slice(5).map((s) => (
                <button
                  key={s.id}
                  onClick={() => onQuestionClick(s.question)}
                  className="flex items-center gap-2 text-left rounded-xl border border-gray-200 bg-white px-3.5 py-3 hover:border-blue-300 hover:shadow-sm transition-all group"
                >
                  <span className="text-base shrink-0">{getIcon(s)}</span>
                  <span className="text-sm text-gray-700 group-hover:text-blue-600 leading-snug">
                    {s.question}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
