import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function Icon({ size = 20, ...props }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    />
  )
}

export function BotIcon(props: IconProps) {
  return (
    <svg width={props.size ?? 20} height={props.size ?? 20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...props}>
      <rect x="3" y="7" width="18" height="13" rx="3" />
      <path d="M8 7V5a4 4 0 0 1 8 0v2" />
      <line x1="8" y1="12" x2="8.01" y2="12" />
      <line x1="12" y1="12" x2="12.01" y2="12" />
      <line x1="16" y1="12" x2="16.01" y2="12" />
    </svg>
  )
}

export function SendIcon(props: IconProps) { return <Icon {...props}><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></Icon> }
export function SparklesIcon(props: IconProps) { return <Icon {...props}><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" /></Icon> }
export function ThumbsUpIcon(props: IconProps) { return <Icon {...props}><path d="M7 10v12" /><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z" /></Icon> }
export function ThumbsDownIcon(props: IconProps) { return <Icon {...props}><path d="M17 14V2" /><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22h0a3.13 3.13 0 0 1-3-3.88Z" /></Icon> }
export function CopyIcon(props: IconProps) { return <Icon {...props}><rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></Icon> }
export function CheckIcon(props: IconProps) { return <Icon {...props}><polyline points="20 6 9 17 4 12" /></Icon> }
export function ChevronDownIcon(props: IconProps) { return <Icon {...props}><polyline points="6 9 12 15 18 9" /></Icon> }
export function ChevronUpIcon(props: IconProps) { return <Icon {...props}><polyline points="18 15 12 9 6 15" /></Icon> }
export function BrainIcon(props: IconProps) { return <Icon {...props}><path d="M12 3a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" /><path d="M9 10a3 3 0 1 0 6 0" /><path d="M12 10v10" /><path d="M9 20a3 3 0 1 0 6 0" /><path d="M9 21h6" /></Icon> }
export function MessageCircleIcon(props: IconProps) { return <Icon {...props}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></Icon> }
export function XIcon(props: IconProps) { return <Icon {...props}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></Icon> }
export function RefreshCwIcon(props: IconProps) { return <Icon {...props}><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></Icon> }
