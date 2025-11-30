'use client'

import { useState } from 'react'
import axios from 'axios'

interface Message {
  role: string
  content: string
  name?: string
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setFinalReport(null)

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

  const checklistItems = Object.values(checklist)
  const completedCount = checklistItems.filter(item => item.status === 'completed').length

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            🔍 Deep Research Agent
          </h1>
          <p className="text-blue-200 text-lg">
            Autonomous AI researcher powered by LangGraph & GPT
          </p>
        </div>

        {/* Search Input */}
        <div className="mb-8">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex gap-3">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to research?"
                className="flex-1 px-6 py-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="px-8 py-4 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold transition-colors text-lg"
              >
                {loading ? 'Researching...' : 'Research'}
              </button>
            </div>
          </form>
        </div>

        {/* Results Grid */}
        {(checklistItems.length > 0 || finalReport) && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Checklist */}
            <div className="lg:col-span-1">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    📋 Checklist
                  </h2>
                  <span className="text-sm text-blue-200">
                    {completedCount}/{checklistItems.length}
                  </span>
                </div>
                
                <div className="space-y-3">
                  {checklistItems.map((item) => (
                    <div
                      key={item.id}
                      className={`p-4 rounded-lg ${
                        item.status === 'completed'
                          ? 'bg-green-500/20 border border-green-500/30'
                          : 'bg-white/5 border border-white/10'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <span className="text-lg">
                          {item.status === 'completed' ? '✅' : '⏳'}
                        </span>
                        <div className="flex-1">
                          <p className="text-white text-sm">{item.question}</p>
                          {item.source_ids.length > 0 && (
                            <p className="text-xs text-blue-300 mt-1">
                              {item.source_ids.length} sources
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Sources Count */}
                <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
                  <p className="text-sm text-blue-200">
                    📚 <strong>{sources.length}</strong> sources found
                  </p>
                </div>
              </div>
            </div>

            {/* Final Report */}
            <div className="lg:col-span-2">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  📊 Research Report
                </h2>
                
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                  </div>
                ) : finalReport ? (
                  <div className="prose prose-invert max-w-none">
                    <div className="text-blue-100 leading-relaxed whitespace-pre-wrap">
                      {finalReport}
                    </div>
                  </div>
                ) : (
                  <p className="text-blue-200 text-center py-12">
                    Research in progress...
                  </p>
                )}
              </div>

              {/* Sources */}
              {sources.length > 0 && (
                <div className="mt-6 bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                  <h3 className="text-lg font-bold text-white mb-4">
                    📚 Sources
                  </h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {sources.map((source) => (
                      <div
                        key={source.id}
                        className="p-4 bg-white/5 rounded-lg border border-white/10"
                      >
                        <p className="text-xs text-blue-300 mb-1">[{source.id}]</p>
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300 font-medium block mb-2"
                        >
                          {source.title}
                        </a>
                        <p className="text-sm text-gray-300 line-clamp-2">
                          {source.content}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Initial State */}
        {checklistItems.length === 0 && !finalReport && !loading && (
          <div className="max-w-3xl mx-auto text-center py-20">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-12 border border-white/20">
              <p className="text-blue-200 text-lg mb-6">
                Enter a research query above to get started
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                <div className="p-4 bg-white/5 rounded-lg">
                  <p className="text-white font-semibold mb-2">1. Query</p>
                  <p className="text-sm text-blue-200">Enter your research question</p>
                </div>
                <div className="p-4 bg-white/5 rounded-lg">
                  <p className="text-white font-semibold mb-2">2. Research</p>
                  <p className="text-sm text-blue-200">Agent searches and analyzes</p>
                </div>
                <div className="p-4 bg-white/5 rounded-lg">
                  <p className="text-white font-semibold mb-2">3. Report</p>
                  <p className="text-sm text-blue-200">Get comprehensive findings</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}

