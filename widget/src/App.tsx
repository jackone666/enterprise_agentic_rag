import ChatWidget from './components/ChatWidget'
import FloatingWidget from './components/FloatingWidget'

/**
 * 应用根组件
 *
 * 根据 URL 参数决定渲染模式：
 * - /?mode=embedded 或包含 widget 参数 → 浮动挂件模式
 * - 默认 → 独立全页模式
 */
export default function App() {
  const params = new URLSearchParams(window.location.search)
  const mode = params.get('mode')

  if (mode === 'embedded' || mode === 'floating') {
    return <FloatingWidget />
  }

  return <ChatWidget mode="page" />
}
