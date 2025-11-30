"""FastAPI backend for the deep research agent."""
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Load configuration first
import config

from research_agent import DeepResearchAgent
from models import AgentContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Research Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
logger.info("🚀 Initializing Deep Research Agent...")
logger.info(f"   Model: {config.DEFAULT_MODEL}")
logger.info(f"   Max iterations: {config.MAX_ITERATIONS}")
try:
    agent = DeepResearchAgent(
        model=config.DEFAULT_MODEL,
        max_iterations=config.MAX_ITERATIONS
    )
    logger.info("✅ Agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize agent: {e}")
    logger.error(traceback.format_exc())
    raise


class ResearchRequest(BaseModel):
    query: str
    messages: List[Dict[str, Any]] = []


class ResearchResponse(BaseModel):
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    final_report: str | None
    needs_clarification: bool = False
    clarifying_questions: List[str] = []


@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """Run research on a query."""
    try:
        logger.info(f"📥 Received research request: {request.query[:100]}...")
        
        # Build messages list
        messages = request.messages if request.messages else [
            {"role": "user", "content": request.query}
        ]
        logger.info(f"📨 Messages count: {len(messages)}")
        
        # Run research
        logger.info("🤖 Starting research agent...")
        result = agent.research(messages)
        logger.info(f"✅ Research complete. Messages: {len(result['messages'])}")
        
        # Check for clarification
        needs_clarification = False
        clarifying_questions = []
        for msg in result["messages"]:
            if msg["role"] == "tool" and msg.get("name") == "ask_clarification":
                needs_clarification = True
                logger.info("❓ Agent requested clarification")
                # Parse questions from message
                content = msg["content"]
                if "1." in content:
                    clarifying_questions = [
                        line.strip()[3:].strip() 
                        for line in content.split('\n') 
                        if line.strip() and line.strip()[0].isdigit()
                    ]
        
        logger.info(f"📊 Final report: {bool(result.get('final_report'))}")
        
        return ResearchResponse(
            messages=result["messages"],
            context=result["context"],
            final_report=result.get("final_report"),
            needs_clarification=needs_clarification,
            clarifying_questions=clarifying_questions
        )
    
    except Exception as e:
        logger.error(f"❌ Error in research endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

