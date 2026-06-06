import type { ChatResponse } from '../types/chat'
import { ShieldAlert, ShieldCheck } from './icons'

export default function FallbackPanel({ response }: { response: ChatResponse | null }) {
  if (!response) {
    return (
      <div className="p-4 text-sm text-gray-400 text-center">
        <ShieldCheck size={24} className="mx-auto mb-2 opacity-50" />
        <p>等待请求...</p>
      </div>
    )
  }

  const hasFallback = !!response.fallback_reason || response.need_human

  if (!hasFallback) {
    return (
      <div className="p-4 text-sm text-center">
        <ShieldCheck size={24} className="mx-auto mb-2 text-green-400" />
        <p className="text-green-700 font-medium">无兜底触发</p>
        <p className="text-xs text-gray-400 mt-1">当前请求正常处理完成</p>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-3 text-sm">
      <div className="flex items-center gap-1.5 font-semibold text-amber-700">
        <ShieldAlert size={14} /> 兜底信息
      </div>

      {/* Fallback reason */}
      {response.fallback_reason && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
          <div className="text-xs font-medium text-amber-800">兜底原因</div>
          <div className="text-xs text-amber-700 mt-1 font-mono">{response.fallback_reason}</div>
        </div>
      )}

      {/* Recovery action */}
      {response.recovery_action && (
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">恢复动作</span>
          <span className="font-mono font-medium text-gray-700">{response.recovery_action}</span>
        </div>
      )}

      {/* Need human */}
      <div className={`rounded-lg border p-3 ${response.need_human ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}`}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">{response.need_human ? '⚠️ 人工兜底已触发' : '✅ 无需人工介入'}</span>
        </div>
        {response.need_human && (
          <div className="text-[10px] text-gray-600 mt-1">
            该问题将进入人工客服队列，客服人员将尽快与您联系
          </div>
        )}
      </div>

      {/* Verification */}
      {response.verification_reason && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
          <div className="text-xs font-medium text-gray-700">校验详情</div>
          <div className="text-[10px] text-gray-600 mt-1">{response.verification_reason}</div>
          <div className={`text-[10px] font-medium mt-1 ${response.verified ? 'text-green-600' : 'text-red-600'}`}>
            {response.verified ? '✅ 校验通过' : '❌ 校验未通过'}
          </div>
        </div>
      )}

      {/* Retry info */}
      {response.retry_history && response.retry_history.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
          <div className="text-xs font-medium text-gray-700 mb-1">重试历史</div>
          {response.retry_history.map((r, i) => {
            const entry = r as Record<string, unknown>
            return (
              <div key={i} className="text-[10px] text-gray-600 font-mono">
                {String(entry.node ?? '?')} → {String(entry.attempt ?? '?')}次
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
