import type { ChatMessage as ChatMessageType } from '../types/chat'
import { BotIcon } from './icons'
import DeepThinking from './DeepThinking'
import FeedbackButtons from './FeedbackButtons'

interface Props {
  msg: ChatMessageType
  sessionId: string
}

/**
 * 聊天消息气泡
 * 模仿华为智能客服样式：用户右侧蓝色气泡，AI 左侧白底+思考链+反馈
 */
export default function ChatMessageBubble({ msg, sessionId }: Props) {
  const isUser = msg.role === 'user'

  return (
    <div className={`msg-enter flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* 头像 */}
      <div
        className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-sm ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gradient-to-br from-gray-100 to-gray-200 text-gray-600'
        }`}
      >
        {isUser ? '👤' : <BotIcon size={16} />}
      </div>

      {/* 内容区 */}
      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {isUser ? (
          /* 用户消息 */
          <div className="rounded-2xl rounded-tr-sm bg-blue-500 text-white px-4 py-2.5 text-sm shadow-sm">
            {msg.content}
          </div>
        ) : (
          /* AI 回复 */
          <div className="space-y-2">
            {/* 深度思考 */}
            {msg.thinking && <DeepThinking content={msg.thinking} complete={msg.complete} />}

            {/* 回答正文 */}
            {msg.content && (
              <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 shadow-sm px-4 py-3">
                <div className="text-sm leading-relaxed whitespace-pre-wrap text-gray-800">
                  {msg.content}
                </div>

                {/* 引用来源 */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-gray-100">
                    <div className="text-[10px] text-gray-400 mb-1">参考来源：</div>
                    <div className="flex flex-wrap gap-1">
                      {msg.citations.map((c, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1 rounded bg-blue-50 px-1.5 py-0.5 text-[10px] text-blue-600"
                        >
                          [{c.index ?? i + 1}] {c.source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 加载中 */}
            {!msg.content && !msg.complete && (
              <div className="rounded-2xl rounded-tl-sm bg-white border border-gray-200 shadow-sm px-4 py-3">
                <div className="flex gap-1.5">
                  <span className="typing-dot w-2 h-2 rounded-full bg-gray-400" />
                  <span className="typing-dot w-2 h-2 rounded-full bg-gray-400" />
                  <span className="typing-dot w-2 h-2 rounded-full bg-gray-400" />
                </div>
              </div>
            )}

            {/* 反馈按钮 */}
            {msg.content && msg.complete && (
              <div className="px-1">
                <FeedbackButtons
                  traceId={sessionId}
                  sessionId={sessionId}
                  content={msg.content}
                />
              </div>
            )}

            {/* AI 免责声明 */}
            {msg.content && msg.complete && (
              <p className="text-[10px] text-gray-300 px-1">内容由AI生成，仅供参考</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
