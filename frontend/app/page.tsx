'use client'

import { useState } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Message {
  role: string
  content: string | null
  name?: string
  tool_calls?: Array<{
    id: string
    function: {
      name: string
      arguments: string
    }
  }>
  tool_call_id?: string
}

interface ChecklistItem {
  id: string
  question: string
  status: 'pending' | 'completed'
  findings?: string
  source_ids: string[]
}

interface Source {
  id: string
  url: string
  title: string
  content: string
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [checklist, setChecklist] = useState<Record<string, ChecklistItem>>({})
  const [sources, setSources] = useState<Source[]>([])
  const [finalReport, setFinalReport] = useState<string | null>(null)
  const [activityLog, setActivityLog] = useState<Array<{type: string, data: any}>>([])
  const [useStreaming, setUseStreaming] = useState(true) // Toggle for streaming vs non-streaming
  const [expandedChecklistItems, setExpandedChecklistItems] = useState<Set<string>>(new Set())
  const [expandedReasoningItems, setExpandedReasoningItems] = useState<Set<number>>(new Set())

  // Helper to render clickable citations
  const renderCitation = (citationNumber: string) => {
    return (
      <a
        href={`#source-${citationNumber}`}
        className="text-blue-400 hover:text-blue-300 font-medium cursor-pointer no-underline"
        onClick={(e) => {
          e.preventDefault()
          document.getElementById(`source-${citationNumber}`)?.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          })
        }}
      >
        <sup>[{citationNumber}]</sup>
      </a>
    )
  }

  // Helper to toggle checklist item expansion
  const toggleChecklistItem = (itemId: string) => {
    setExpandedChecklistItems(prev => {
      const newSet = new Set(prev)
      if (newSet.has(itemId)) {
        newSet.delete(itemId)
      } else {
        newSet.add(itemId)
      }
      return newSet
    })
  }


  const handleSubmitStreaming = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setFinalReport(null)
    setActivityLog([])
    setChecklist({})
    setSources([])
    setExpandedChecklistItems(new Set())
    setExpandedReasoningItems(new Set())

    try {
      const eventSource = new EventSource(
        `http://localhost:8000/api/research/stream?${new URLSearchParams({
          query,
        })}`
      )

      // We'll send query via POST body instead, so we need fetch
      eventSource.close()

      // Use fetch for POST with SSE
      const response = await fetch('http://localhost:8000/api/research/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, messages: [] }),
      })

      if (!response.ok) {
        throw new Error('Failed to start research')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''
      let currentEventType = 'unknown'

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEventType = line.substring(6).trim()
          } else if (line.startsWith('data:')) {
            try {
              const data = JSON.parse(line.substring(5).trim())
              handleStreamEvent(currentEventType, data)
              currentEventType = 'unknown' // Reset after handling
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }

    } catch (error) {
      console.error('Error:', error)
      alert('Error running research. Make sure the API is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleStreamEvent = (eventType: string, data: any) => {
    console.log('Stream event:', eventType, data)

    switch (eventType) {
      case 'agent_reasoning':
        setActivityLog(prev => [...prev, {
          type: 'reasoning',
          data: { 
            content: data.content
          }
        }])
        break

      case 'tool_call_started':
        setActivityLog(prev => [...prev, {
          type: 'started',
          data: { 
            tool_name: data.tool_name, 
            tool_call_id: data.tool_call_id,
            arguments: data.arguments 
          }
        }])
        break

      case 'tool_call_completed':
        setActivityLog(prev => [...prev, {
          type: 'completed',
          data: { 
            tool_name: data.tool_name, 
            success: data.success 
          }
        }])
        break

      case 'checklist_updated':
        setChecklist(data.items)
        break

      case 'source_discovered':
        setSources(prev => [...prev, ...data.sources])
        break

      case 'final_report':
        setFinalReport(data.report)
        break

      case 'complete':
        setMessages(data.messages || [])
        if (data.context?.checklist) {
          setChecklist(data.context.checklist)
        }
        if (data.context?.sources) {
          setSources(data.context.sources)
        }
        break

      case 'error':
        console.error('Stream error:', data.error)
        alert(`Error: ${data.error}`)
        break
    }
  }

  const handleSubmitNonStreaming = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setFinalReport(null)
    setActivityLog([])
    setExpandedChecklistItems(new Set())
    setExpandedReasoningItems(new Set())

    try {
      const response = await axios.post('http://localhost:8000/api/research', {
        query,
        messages: [],
      })

      setMessages(response.data.messages)
      setChecklist(response.data.context.checklist)
      setSources(response.data.context.sources)
      setFinalReport(response.data.final_report)
    } catch (error) {
      console.error('Error:', error)
      alert('Error running research. Make sure the API is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = useStreaming ? handleSubmitStreaming : handleSubmitNonStreaming

  const checklistItems = Object.values(checklist)
  const completedCount = checklistItems.filter(item => item.status === 'completed').length

  return (
    <main className="min-h-screen bg-gray-900">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            🔍 Deep Research Agent
          </h1>
          <p className="text-gray-400 text-lg">
            Autonomous AI researcher powered by LangGraph & GPT
          </p>
        </div>

        {/* Search Input */}
        <div className="mb-8">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex gap-3 mb-3">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to research?"
                className="flex-1 px-6 py-4 rounded-xl bg-gray-800 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="px-8 py-4 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold transition-colors text-lg"
              >
                {loading ? 'Researching...' : 'Research'}
              </button>
            </div>
            <div className="flex justify-center">
              <button
                type="button"
                onClick={() => setUseStreaming(!useStreaming)}
                className="text-sm text-gray-400 hover:text-gray-300 transition-colors"
              >
                {useStreaming ? '🔄 Real-time updates (SSE)' : '⏸️ Wait for results'} • Click to toggle
              </button>
            </div>
          </form>
        </div>

        {/* Final Report - Main Focus */}
        {finalReport && (
          <div className="mb-8">
            <div className="bg-gray-800 rounded-xl p-10 border border-gray-700">
              <h2 className="text-3xl font-bold text-white mb-8 pb-4 border-b border-gray-700">
                📊 Research Report
              </h2>
              
              <div className="prose prose-lg prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({ node, ...props }) => <h1 className="text-3xl font-bold text-white mt-8 mb-4" {...props} />,
                    h2: ({ node, ...props }) => <h2 className="text-2xl font-bold text-white mt-10 mb-4 pb-2 border-b border-gray-700" {...props} />,
                    h3: ({ node, ...props }) => <h3 className="text-xl font-semibold text-gray-200 mt-8 mb-3" {...props} />,
                    h4: ({ node, ...props }) => <h4 className="text-lg font-medium text-gray-300 mt-6 mb-2" {...props} />,
                    p: ({ node, children, ...props }) => {
                      // Recursively process children to convert [1] citations to clickable links
                      const processCitations = (child: any): any => {
                        if (typeof child === 'string') {
                          // Split by citation pattern [N]
                          const parts = child.split(/(\[\d+\])/)
                          return parts.map((part, i) => {
                            const match = part.match(/\[(\d+)\]/)
                            if (match) {
                              return <span key={i}>{renderCitation(match[1])}</span>
                            }
                            return part
                          })
                        }
                        if (Array.isArray(child)) {
                          return child.map(processCitations)
                        }
                        return child
                      }
                      
                      const processedChildren = Array.isArray(children) 
                        ? children.map(processCitations)
                        : processCitations(children)
                      
                      return <p className="my-5 text-gray-300 leading-relaxed" {...props}>{processedChildren}</p>
                    },
                    ul: ({ node, ...props }) => <ul className="list-disc list-inside my-4 space-y-2" {...props} />,
                    ol: ({ node, ...props }) => <ol className="list-decimal list-inside my-4 space-y-2" {...props} />,
                    li: ({ node, children, ...props }) => {
                      // Process citations in list items too
                      const processCitations = (child: any): any => {
                        if (typeof child === 'string') {
                          const parts = child.split(/(\[\d+\])/)
                          return parts.map((part, i) => {
                            const match = part.match(/\[(\d+)\]/)
                            if (match) {
                              return <span key={i}>{renderCitation(match[1])}</span>
                            }
                            return part
                          })
                        }
                        if (Array.isArray(child)) {
                          return child.map(processCitations)
                        }
                        return child
                      }
                      
                      const processedChildren = Array.isArray(children) 
                        ? children.map(processCitations)
                        : processCitations(children)
                      
                      return <li className="text-gray-300 ml-4" {...props}>{processedChildren}</li>
                    },
                    strong: ({ node, ...props }) => <strong className="text-white font-semibold" {...props} />,
                    em: ({ node, ...props }) => <em className="text-gray-200 italic" {...props} />,
                    a: ({ node, ...props }) => <a className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer" {...props} />,
                    blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-gray-600 pl-4 italic text-gray-400 my-4" {...props} />,
                    code: ({ node, inline, ...props }: any) => 
                      inline 
                        ? <code className="bg-gray-700 px-2 py-1 rounded text-sm text-gray-200" {...props} />
                        : <code className="block bg-gray-700 p-4 rounded text-sm text-gray-200 overflow-x-auto" {...props} />,
                  }}
                >
                  {finalReport}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}

        {/* Supporting Information Grid */}
        {(checklistItems.length > 0 || finalReport || loading) && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Checklist */}
            <div className="lg:col-span-1">
              <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    📋 Research Plan
                  </h2>
                  <span className="text-xs text-gray-400 font-medium">
                    {completedCount}/{checklistItems.length}
                  </span>
                </div>
                
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {checklistItems.length === 0 ? (
                    <p className="text-gray-500 text-xs text-center py-4">No items yet</p>
                  ) : (
                    checklistItems.map((item, idx) => (
                      <div
                        key={item.id}
                        className={`rounded-lg border ${
                          item.status === 'completed'
                            ? 'bg-green-900/20 border-green-700'
                            : 'bg-gray-700/30 border-gray-600'
                        }`}
                      >
                        {/* Item header - clickable if has findings */}
                        <div
                          onClick={() => item.findings && toggleChecklistItem(item.id)}
                          className={`p-2 flex items-start gap-2 ${
                            item.findings ? 'cursor-pointer hover:bg-gray-700/30' : ''
                          }`}
                        >
                          <span className="text-xs flex-shrink-0">
                            {item.status === 'completed' ? '✅' : '⏳'}
                          </span>
                          <p className="text-gray-300 text-xs leading-relaxed flex-1">
                            {item.question}
                          </p>
                          {item.findings && (
                            <span className="text-gray-500 text-xs flex-shrink-0">
                              {expandedChecklistItems.has(item.id) ? '▼' : '▶'}
                            </span>
                          )}
                        </div>

                        {/* Expandable findings */}
                        {item.findings && expandedChecklistItems.has(item.id) && (
                          <div className="px-2 pb-2 border-t border-gray-600/50 mt-1">
                            <div className="mt-2 bg-gray-900/40 rounded p-2 border border-gray-700">
                              <p className="text-[10px] font-semibold text-gray-400 mb-1">
                                Findings:
                              </p>
                              <div className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
                                {item.findings}
                              </div>
                            </div>
                            {item.source_ids && item.source_ids.length > 0 && (
                              <div className="mt-2">
                                <p className="text-[10px] text-gray-500 mb-1">
                                  Sources: {item.source_ids.length}
                                </p>
                                <div className="flex flex-wrap gap-1">
                                  {item.source_ids.map((sourceId) => (
                                    <a
                                      key={sourceId}
                                      href={`#source-${sourceId.replace('source_', '')}`}
                                      onClick={(e) => {
                                        e.preventDefault()
                                        const id = sourceId.replace('source_', '')
                                        document.getElementById(`source-${id}`)?.scrollIntoView({
                                          behavior: 'smooth',
                                          block: 'center'
                                        })
                                      }}
                                      className="text-blue-400 hover:text-blue-300 text-[10px] font-mono"
                                    >
                                      [{sourceId.replace('source_', '')}]
                                    </a>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Activity Log */}
            <div className="lg:col-span-1">
              <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <h2 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                  🔧 Activity Log
                </h2>
                
                <div className="space-y-1.5 max-h-64 overflow-y-auto">
                  {useStreaming && activityLog.length > 0 ? (
                    // Show real-time activity log from streaming
                    activityLog.map((activity, idx) => {
                      // Handle agent reasoning
                      if (activity.type === 'reasoning') {
                        const isExpanded = expandedReasoningItems.has(idx)
                        const content = activity.data.content
                        const preview = content.length > 80 ? content.substring(0, 80) + '...' : content
                        
                        return (
                          <div 
                            key={idx}
                            className="rounded border bg-purple-900/30 border-purple-700"
                          >
                            <div 
                              onClick={() => {
                                const newSet = new Set(expandedReasoningItems)
                                if (isExpanded) {
                                  newSet.delete(idx)
                                } else {
                                  newSet.add(idx)
                                }
                                setExpandedReasoningItems(newSet)
                              }}
                              className="p-2 cursor-pointer hover:bg-purple-900/40 transition-colors"
                            >
                              <div className="flex items-start gap-2">
                                <span className="text-xs text-purple-300">💭</span>
                                <p className="text-xs text-purple-200 flex-1 leading-relaxed">
                                  {isExpanded ? content : preview}
                                </p>
                                {content.length > 80 && (
                                  <span className="text-purple-500 text-xs flex-shrink-0">
                                    {isExpanded ? '▼' : '▶'}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        )
                      }
                      
                      // Parse arguments if it's a JSON string
                      let parsedArgs = null
                      if (activity.type === 'started' && activity.data.arguments) {
                        try {
                          parsedArgs = JSON.parse(activity.data.arguments)
                        } catch (e) {
                          parsedArgs = activity.data.arguments
                        }
                      }

                      return (
                        <div 
                          key={idx}
                          className={`p-2 rounded border ${
                            activity.type === 'started' 
                              ? 'bg-blue-900/40 border-blue-700'
                              : activity.data.success
                              ? 'bg-green-900/40 border-green-700'
                              : 'bg-red-900/40 border-red-700'
                          }`}
                        >
                          <p className="text-xs font-mono">
                            {activity.type === 'started' ? (
                              <>
                                <span className="text-blue-300">🔧 {activity.data.tool_name}</span>
                                {parsedArgs && (
                                  <span className="text-blue-200/60 block mt-1 text-[10px] leading-tight">
                                    {typeof parsedArgs === 'object' 
                                      ? Object.entries(parsedArgs).map(([key, value]) => {
                                          // Truncate long values
                                          const displayValue = typeof value === 'string' && value.length > 50
                                            ? value.substring(0, 50) + '...'
                                            : JSON.stringify(value)
                                          return `${key}: ${displayValue}`
                                        }).join(', ')
                                      : String(parsedArgs)
                                    }
                                  </span>
                                )}
                              </>
                          ) : (
                            <span className={activity.data.success ? 'text-green-300' : 'text-red-300'}>
                              {activity.data.success ? '✓' : '✗'} {activity.data.tool_name}
                            </span>
                          )}
                        </p>
                      </div>
                      )
                    })
                  ) : (
                    // Fallback to message-based activity log
                    messages.map((msg, idx) => {
                      if (msg.role === 'assistant' && msg.tool_calls) {
                        return msg.tool_calls.map((tc, tcIdx) => (
                          <div key={`${idx}-${tcIdx}`} className="p-2 bg-blue-900/40 rounded border border-blue-700">
                            <p className="text-xs text-blue-300 font-mono">
                              🤖 {tc.function.name}
                            </p>
                          </div>
                        ))
                      }
                      if (msg.role === 'tool' && msg.name) {
                        return (
                          <div key={idx} className="p-2 bg-green-900/40 rounded border border-green-700">
                            <p className="text-xs text-green-300 font-mono">
                              ✓ {msg.name}
                            </p>
                          </div>
                        )
                      }
                      return null
                    })
                  )}
                </div>
              </div>
            </div>

            {/* Sources */}
            <div className="lg:col-span-1">
              {loading ? (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                  <div className="flex flex-col items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
                    <p className="text-gray-400 text-sm">Researching...</p>
                  </div>
                </div>
              ) : sources.length > 0 ? (
                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
                  <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                    📚 Sources ({sources.length})
                  </h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {sources.map((source) => (
                      <div
                        key={source.id}
                        id={`source-${source.id}`}
                        className="p-3 bg-gray-700/50 rounded-lg border border-gray-600 hover:bg-gray-700 transition-colors scroll-mt-4"
                      >
                        <p className="text-xs text-gray-400 mb-1 font-mono">[{source.id}]</p>
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300 font-medium text-xs block mb-1 line-clamp-2"
                        >
                          {source.title}
                        </a>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* Initial State */}
        {checklistItems.length === 0 && !finalReport && !loading && (
          <div className="max-w-3xl mx-auto text-center py-20">
            <div className="bg-gray-800 rounded-xl p-12 border border-gray-700">
              <p className="text-gray-400 text-lg mb-6">
                Enter a research query above to get started
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                  <p className="text-white font-semibold mb-2">1. Query</p>
                  <p className="text-sm text-gray-400">Enter your research question</p>
                </div>
                <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                  <p className="text-white font-semibold mb-2">2. Research</p>
                  <p className="text-sm text-gray-400">Agent searches and analyzes</p>
                </div>
                <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
                  <p className="text-white font-semibold mb-2">3. Report</p>
                  <p className="text-sm text-gray-400">Get comprehensive findings</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}

