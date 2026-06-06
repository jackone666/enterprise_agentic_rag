import { useState, useRef, useEffect } from 'react'
import { MessageCircleIcon, XIcon } from './icons'
import ChatWidget from './ChatWidget'

/**
 * 浮动挂件 — 嵌入任意网页右下角的聊天入口
 *
 * 用法：
 * ```html
 * <div id="enterprise-rag-widget"></div>
 * <script src="http://localhost:5174/widget.js"></script>
 * ```
 */
export default function FloatingWidget() {
  const [open, setOpen] = useState(false)
  const [unread, setUnread] = useState(false)
  const initialOpen = useRef(false)

  // 标记未读（首次加载时微提醒）
  useEffect(() => {
    if (!initialOpen.current) {
      const timer = setTimeout(() => setUnread(true), 3000)
      initialOpen.current = true
      return () => clearTimeout(timer)
    }
  }, [])

  function handleOpen() {
    setOpen(true)
    setUnread(false)
  }

  return (
    <>
      {/* 浮动按钮 */}
      {!open && (
        <button
          onClick={handleOpen}
          className="fixed bottom-6 right-6 z-[9999] w-14 h-14 rounded-full bg-accent text-white shadow-lg hover:bg-accent-hover hover:scale-110 active:scale-95 transition-all flex items-center justify-center animate-in zoom-in"
          aria-label="打开智能客服"
        >
          <MessageCircleIcon size={24} />
          {unread && (
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-500 border-2 border-white flex items-center justify-center text-[9px] font-bold">
              1
            </span>
          )}
        </button>
      )}

      {/* 聊天面板 */}
      {open && (
        <div className="fixed bottom-6 right-6 z-[9999] w-[400px] h-[600px] rounded-2xl shadow-2xl border border-gray-200 overflow-hidden bg-white flex flex-col animate-in slide-in-from-bottom-4 duration-300">
          {/* 关闭按钮 */}
          <button
            onClick={() => setOpen(false)}
            className="absolute top-3 right-3 z-10 w-7 h-7 rounded-full bg-black/20 hover:bg-black/40 text-white flex items-center justify-center transition-colors"
            aria-label="关闭"
          >
            <XIcon size={14} />
          </button>

          <ChatWidget mode="embedded" />
        </div>
      )}
    </>
  )
}
