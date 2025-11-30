# Deep Research Agent

An autonomous AI research agent that conducts comprehensive research on any topic using web search and LLM-powered analysis.

## Features

- 🤖 **Autonomous Research**: Agent decides what to research and how
- 🔍 **Web Search**: Google search via Serper.dev
- 📋 **Dynamic Checklist**: Creates and tracks research items
- 📊 **Comprehensive Reports**: Synthesizes findings with citations
- 💬 **Clean Interface**: Beautiful Next.js + Tailwind UI
- ⚡ **Real-time Updates**: Watch research progress live

## Architecture

**Backend**: Python + FastAPI + LangGraph + GPT-4  
**Frontend**: Next.js + TypeScript + Tailwind CSS

**Agent Design**: Two-node ReAct loop
- **Agent Node**: LLM decides which tools to use
- **Tools Node**: Executes search, checklist, and reporting tools

## Quick Start

### Prerequisites
- **Python 3.11 or 3.12** (required - the script will check)
- Node.js 18+
- OpenAI API key (https://platform.openai.com/)
- Serper API key (https://serper.dev/)

### Run Locally

**1. Start Backend (Terminal 1)**
```bash
./run-backend.sh
```
First time it will:
- Check for Python 3.11+
- Create virtual environment in `backend/venv/`
- Create `backend/.env` file and pause for you to add API keys:
  ```
  OPENAI_API_KEY=sk-...
  SERPER_API_KEY=...
  ```
- Install all dependencies
- Start the server

Press Enter after adding your API keys to `backend/.env`.

**2. Start Frontend (Terminal 2)**
```bash
./run-frontend.sh
```

**3. Open Browser**
```
http://localhost:3000
```

That's it! The scripts handle all dependencies automatically.

## How It Works

1. **Enter Query**: Type your research question
2. **Agent Plans**: Creates a checklist of things to research
3. **Autonomous Search**: Searches web, gathers sources
4. **Synthesis**: Writes comprehensive report with citations

## Project Structure

```
deep-research/
├── backend/               # Python backend
│   ├── api.py            # FastAPI server
│   ├── research_agent.py # Main agent (LangGraph)
│   ├── tools.py          # 5 agent tools
│   ├── tool_handlers.py  # Tool execution logic
│   ├── models.py         # Pydantic models & state
│   ├── prompts.py        # LLM prompts
│   └── requirements.txt  # Python deps
├── frontend/             # Next.js app
│   ├── app/
│   │   ├── page.tsx     # Main UI
│   │   └── globals.css  # Tailwind styles
│   └── package.json     # Node deps
├── run-backend.sh        # Start backend
└── run-frontend.sh       # Start frontend
```

## Development

### Python API Only
```python
from research_agent import DeepResearchAgent

agent = DeepResearchAgent(model="gpt-4")
messages = [{"role": "user", "content": "Your question"}]
result = agent.research(messages)

print(result["final_report"])
```

### Change Model
Edit `backend/api.py` line 18:
```python
agent = DeepResearchAgent(model="gpt-4")  # or "claude-3-5-sonnet", etc.
```

## Output Structure

The agent returns a dictionary with a **messages** field containing the conversation history:

```python
{
    "messages": [
        {"role": "user", "content": "User's query"},
        {"role": "system", "content": "Internal state updates"},
        {"role": "assistant", "content": "Research plan or clarification questions"},
        {"role": "tool", "content": "Search results", "name": "web_search"},
        {"role": "tool", "content": "Summaries", "name": "search_results"},
        {"role": "assistant", "content": "Final comprehensive report"}
    ],
    "needs_clarification": False,  # True if waiting for user input
    "clarifying_questions": [],    # List of questions if clarification needed
    "plan": [...],                 # Research sub-questions
    "current_step": 3              # Progress tracker
}
```

**Message Roles:**
- `user`: User's input queries and responses
- `assistant`: Agent's responses (clarifications, plan, final report)
- `tool`: Search operations and results
- `system`: Internal state information

## Available Tools

The agent has 5 tools at its disposal:

1. **search(query)** - Search the web via Serper
2. **ask_clarification(questions)** - Ask user for clarification
3. **modify_checklist(items)** - Create/update research plan
4. **write_subreport(item_id, findings, sources)** - Document findings
5. **finish(final_report)** - Complete with final report

## Tech Stack

- **Agent**: LangGraph + LiteLLM
- **Backend**: FastAPI + Python
- **Frontend**: Next.js 14 + TypeScript + Tailwind
- **Search**: Serper.dev (Google Search API)
- **LLM**: GPT-4 (configurable to Claude, Gemini, etc.)

## How It Works

### 1. Clarification Phase (Optional)
- GPT-5 analyzes the query for clarity
- Determines if additional context would improve research quality
- Generates 2-3 specific clarifying questions if needed
- Incorporates user responses into enhanced query

### 2. Planning Phase
The agent sends your question to GPT-5 and asks it to break it down into focused sub-questions.

### 3. Research Phase
For each sub-question:
- Searches Google using Serper.dev API
- Retrieves top 3 most relevant results
- Uses GPT-5 to summarize the findings

### 4. Synthesis Phase
- Combines all findings from previous steps
- Uses GPT-5 to write a comprehensive report
- Integrates evidence and draws conclusions

## Example Output

### Scenario 1: Query needs clarification
```
👤 USER: Tell me about AI

🤖 ASSISTANT (Clarification):
I need some clarification to provide better research:

1. What specific aspect of AI are you interested in (e.g., machine learning, NLP, computer vision)?
2. What timeframe are you interested in (e.g., historical overview, recent developments, future trends)?
3. Are you looking for technical details or general applications?

Please provide answers to help me focus the research.

👤 USER: I'm interested in large language models from 2024, business use cases

🤖 ASSISTANT (Plan):
Research Plan:
1. What are the major LLM releases in 2024?
2. What are successful enterprise applications?
3. What are the ROI and implementation challenges?

🔧 TOOL (web_search): Search query: What are the major LLM releases in 2024?
🔧 TOOL (search_results): Sources: [URLs]
Summary: [Key findings]

[... more research steps ...]

🤖 ASSISTANT (Final Report):
[Comprehensive synthesized report based on all research]
```

### Scenario 2: Specific query, no clarification needed
```
👤 USER: What are the latest developments in quantum computing hardware in 2024?

🤖 ASSISTANT (Plan):
Research Plan:
1. What are the most recent breakthroughs in quantum computing hardware?
2. What new quantum algorithms have been developed?
3. What are the current commercial applications of quantum computing?

🔧 TOOL: web_search
🔧 TOOL: search_results
[... research execution ...]

🤖 ASSISTANT (Final Report):
================================================================================
Quantum computing in 2024 has seen significant advances across hardware,
algorithms, and commercial applications...

[Comprehensive report with citations and evidence from web search]
================================================================================
```

## Troubleshooting

**Backend won't start**: Make sure API keys are set in `.env`  
**Frontend can't connect**: Ensure backend is running on port 8000  
**Dependencies fail**: Try `pip install --upgrade pip` or `npm cache clean --force`

## License

MIT

