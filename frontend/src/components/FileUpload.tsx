import { useState, useRef, useCallback } from 'react'
import { Upload, X, FileText, CheckCircle } from './icons'

interface UploadResult {
  success: boolean
  filename: string
  chunks: number
  milvus_indexed: number
  sections: number
  tables: number
  minio_uploaded: boolean
}

export default function FileUpload() {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState<UploadResult[]>([])
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files).filter(
      f => f.name.endsWith('.md') || f.name.endsWith('.txt') || f.name.endsWith('.pdf') || f.name.endsWith('.docx')
    )
    setFiles(prev => [...prev, ...dropped])
  }, [])

  async function uploadAll() {
    if (files.length === 0) return
    setUploading(true)
    setError('')
    const newResults: UploadResult[] = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      try {
        const form = new FormData()
        form.append('file', file)
        const resp = await fetch('/admin/upload', { method: 'POST', body: form })
        if (resp.ok) {
          newResults.push(await resp.json())
        } else {
          setError(`${file.name}: HTTP ${resp.status}`)
        }
      } catch {
        setError(`${file.name}: 上传失败`)
      }
    }

    setResults(prev => [...prev, ...newResults])
    setFiles([])
    setUploading(false)
  }

  function removeFile(name: string) {
    setFiles(prev => prev.filter(f => f.name !== name))
  }

  return (
    <div className="p-4 text-sm">
      <div className="flex items-center gap-1.5 font-semibold text-gray-700 mb-3">
        <Upload size={14} /> 文档上传
      </div>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-colors"
      >
        <Upload size={24} className="mx-auto text-gray-400 mb-1" />
        <p className="text-xs text-gray-500">拖拽文件到此处或点击上传</p>
        <p className="text-[10px] text-gray-400 mt-0.5">支持 .md .txt .pdf .docx</p>
        <input ref={inputRef} type="file" className="hidden" multiple accept=".md,.txt,.pdf,.docx"
          onChange={e => setFiles(prev => [...prev, ...Array.from(e.target.files || [])])} />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-3 space-y-1">
          <div className="text-xs font-medium text-gray-600">待上传 ({files.length})</div>
          {files.map(f => (
            <div key={f.name} className="flex items-center justify-between bg-gray-50 rounded px-2 py-1.5 text-xs">
              <span className="flex items-center gap-1.5">
                <FileText size={12} className="text-blue-400" />
                <span className="truncate max-w-[200px]">{f.name}</span>
                <span className="text-gray-400">{(f.size / 1024).toFixed(0)}KB</span>
              </span>
              <button onClick={() => removeFile(f.name)} className="text-gray-400 hover:text-red-500">
                <X size={14} />
              </button>
            </div>
          ))}
          <button
            onClick={uploadAll}
            disabled={uploading}
            className="w-full mt-2 rounded-lg bg-blue-500 text-white py-2 text-xs font-medium hover:bg-blue-600 disabled:opacity-50"
          >
            {uploading ? '上传中...' : `上传 ${files.length} 个文件`}
          </button>
        </div>
      )}

      {error && <div className="mt-2 text-xs text-red-500">{error}</div>}

      {/* Results */}
      {results.length > 0 && (
        <div className="mt-3 space-y-1">
          <div className="text-xs font-medium text-gray-600">上传结果 ({results.length})</div>
          {results.map((r, i) => (
            <div key={i} className="rounded border border-green-200 bg-green-50 p-2 text-xs space-y-0.5">
              <div className="flex items-center gap-1 text-green-700 font-medium">
                <CheckCircle size={12} /> {r.filename}
              </div>
              <div className="text-gray-600">
                {r.chunks} chunks → Milvus {r.milvus_indexed} | MinIO {r.minio_uploaded ? '✅' : '❌'}
                {r.sections > 0 && ` | ${r.sections} 章节`}
                {r.tables > 0 && ` | ${r.tables} 表格`}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
