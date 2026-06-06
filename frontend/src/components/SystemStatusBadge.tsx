import { useEffect, useState } from 'react'
import { Wifi, WifiOff } from './icons'

export default function SystemStatusBadge() {
  const [ok, setOk] = useState<boolean | null>(null)

  useEffect(() => {
    let alive = true
    async function check() {
      try {
        const res = await fetch('/health')
        if (alive) setOk(res.ok)
      } catch {
        if (alive) setOk(false)
      }
    }
    check()
    const iv = setInterval(check, 30000)
    return () => { alive = false; clearInterval(iv) }
  }, [])

  if (ok === null) return <div className="text-xs text-gray-400">检查中...</div>

  return (
    <div className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${
      ok ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'
    }`}>
      {ok ? <Wifi size={12} /> : <WifiOff size={12} />}
      {ok ? '系统正常' : '系统离线'}
    </div>
  )
}
