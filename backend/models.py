"""Pydantic models and type definitions for the agentic deep research agent."""
from typing import List, Optional, Dict, TypedDict, Annotated, Literal, TypeAlias, Any
import operator
from pydantic import BaseModel, Field


# ============================================================================
# TypedDict definitions for LangGraph state
# ============================================================================

class Message(TypedDict, total=False):
    """Standard message format compatible with OpenAI Chat API.
    
    Message history is stored in OpenAI-compatible format for simplicity:
    - User messages: role="user", content="..."
    - Assistant text: role="assistant", content="..."
    - Assistant with tools: role="assistant", content=None, tool_calls=[...]
    - Tool results: role="tool", content="...", tool_call_id="..."
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str]  # Can be None for assistant messages with tool_calls
    name: Optional[str]  # Tool name for tool messages
    tool_calls: Optional[List[Dict[str, Any]]]  # For assistant messages calling tools
    tool_call_id: Optional[str]  # For tool result messages


# Type alias for context dict structure
ContextDict: TypeAlias = Dict[str, Any]  # Serialized AgentContext


class AgentState(TypedDict):
    """LangGraph state for the agentic research agent.
    
    Message History:
    - Stored in OpenAI-compatible format for simplicity
    - Tool calls are preserved in their original structure
    - Easy to pass directly to LLM APIs
    
    Context:
    - Stored as dict for LangGraph compatibility
    - Use AgentContext.from_dict() to load as Pydantic model
    """
    messages: Annotated[list[Message], operator.add]  # Message history
    context: ContextDict  # Agent context (serialized AgentContext)
    final_report: Optional[str]  # Final synthesized report


# ============================================================================
# Pydantic models for data structures
# ============================================================================

class Source(BaseModel):
    """A web source discovered during research."""
    id: str = Field(description="Unique identifier")
    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    content: str = Field(description="Relevant content/snippet")


class ChecklistItem(BaseModel):
    """A research checklist item."""
    id: str = Field(description="Item ID")
    question: str = Field(description="Research question")
    status: Literal["pending", "in_progress", "completed"] = Field(default="pending")
    findings: Optional[str] = Field(default=None, description="Research findings")
    source_ids: List[str] = Field(default_factory=list, description="IDs of sources used")


class Checklist(BaseModel):
    """Research checklist with helper methods."""
    items: Dict[str, ChecklistItem] = Field(default_factory=dict, description="Checklist items by ID")
    
    def add_items(self, questions: List[str]) -> List[str]:
        """Add new items to checklist. Returns list of new item IDs."""
        new_ids = []
        for question in questions:
            item_id = f"item_{len(self.items) + 1}"
            self.items[item_id] = ChecklistItem(
                id=item_id,
                question=question,
                status="pending"
            )
            new_ids.append(item_id)
        return new_ids
    
    def complete_item(self, item_id: str, findings: str, source_ids: List[str]) -> bool:
        """Mark an item as completed with findings. Returns True if successful."""
        if item_id not in self.items:
            return False
        
        self.items[item_id].status = "completed"
        self.items[item_id].findings = findings
        self.items[item_id].source_ids = source_ids
        return True
    
    def get_item(self, item_id: str) -> Optional[ChecklistItem]:
        """Get a checklist item by ID."""
        return self.items.get(item_id)
    
    def get_pending(self) -> List[ChecklistItem]:
        """Get all pending items."""
        return [item for item in self.items.values() if item.status == "pending"]
    
    def get_completed(self) -> List[ChecklistItem]:
        """Get all completed items."""
        return [item for item in self.items.values() if item.status == "completed"]
    
    def all_completed(self) -> bool:
        """Check if all items are completed."""
        return all(item.status == "completed" for item in self.items.values())
    
    def format_display(self) -> str:
        """Format checklist for display."""
        if not self.items:
            return "No checklist items yet"
        
        lines = []
        for item in self.items.values():
            status_icon = {"pending": "☐", "in_progress": "⧗", "completed": "✓"}
            icon = status_icon.get(item.status, "☐")
            lines.append(f"{icon} [{item.id}] {item.question}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dict for state storage."""
        return {item_id: item.model_dump() for item_id, item in self.items.items()}
    
    @classmethod
    def from_dict(cls, data: dict) -> "Checklist":
        """Create from dict."""
        items = {item_id: ChecklistItem(**item_data) for item_id, item_data in data.items()}
        return cls(items=items)


class AgentContext(BaseModel):
    """Context that lives with the agent and is accessible to all tools."""
    checklist: Checklist = Field(default_factory=Checklist)
    sources: List[Source] = Field(default_factory=list)
    source_counter: int = Field(default=0, description="Counter for generating source IDs")
    
    def add_sources(self, search_results: List[dict]) -> List[Source]:
        """Add new sources from search results. Returns list of new Source objects."""
        new_sources = []
        for result in search_results:
            self.source_counter += 1
            source = Source(
                id=f"source_{self.source_counter}",
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", "")
            )
            self.sources.append(source)
            new_sources.append(source)
        return new_sources
    
    def find_source_ids_by_urls(self, urls: List[str]) -> List[str]:
        """Find source IDs for given URLs."""
        return [src.id for src in self.sources if src.url in urls]
    
    def get_sources_count(self) -> int:
        """Get total number of sources."""
        return len(self.sources)
    
    def to_dict(self) -> dict:
        """Convert to dict for state storage."""
        return {
            "checklist": self.checklist.to_dict(),
            "sources": [src.model_dump() for src in self.sources],
            "source_counter": self.source_counter
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentContext":
        """Create from dict."""
        return cls(
            checklist=Checklist.from_dict(data.get("checklist", {})),
            sources=[Source(**s) for s in data.get("sources", [])],
            source_counter=data.get("source_counter", 0)
        )



