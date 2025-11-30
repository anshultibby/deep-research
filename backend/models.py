"""Pydantic models and type definitions for the agentic deep research agent."""
from typing import List, Optional, Dict, TypedDict, Annotated, Literal, TypeAlias, Any
import operator
from pydantic import BaseModel, Field


# ============================================================================
# Constants
# ============================================================================

class StateKeys:
    """Constants for state dictionary keys to avoid magic strings."""
    MESSAGES = "messages"
    CONTEXT = "context"
    FINAL_REPORT = "final_report"


class MessageRoles:
    """Constants for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolNames:
    """Constants for tool names."""
    SEARCH = "search"
    ASK_CLARIFICATION = "ask_clarification"
    GET_CURRENT_CHECKLIST = "get_current_checklist"
    MODIFY_CHECKLIST = "modify_checklist"
    WRITE_SUBREPORT = "write_subreport"
    WRITE_FINAL_REPORT = "write_final_report"


# ============================================================================
# TypedDict definitions for LangGraph state
# ============================================================================

class Message(BaseModel):
    """Standard message format compatible with OpenAI Chat API.
    
    Message history is stored in OpenAI-compatible format for simplicity:
    - User messages: role="user", content="..."
    - Assistant text: role="assistant", content="..."
    - Assistant with tools: role="assistant", content=None, tool_calls=[...]
    - Tool results: role="tool", content="...", tool_call_id="..."
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None  # Can be None for assistant messages with tool_calls
    name: Optional[str] = None  # Tool name for tool messages
    tool_calls: Optional[List[Dict[str, Any]]] = None  # For assistant messages calling tools
    tool_call_id: Optional[str] = None  # For tool result messages

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRoles.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant text message."""
        return cls(role=MessageRoles.ASSISTANT, content=content)
    
    @classmethod
    def assistant_with_tools(cls, tool_calls: List[Dict[str, Any]]) -> "Message":
        """Create an assistant message with tool calls."""
        return cls(
            role=MessageRoles.ASSISTANT,
            content=None,  # No text content when calling tools
            tool_calls=tool_calls
        )
    
    @classmethod
    def tool(cls, content: str, tool_name: str, tool_call_id: str) -> "Message":
        """Create a tool result message."""
        return cls(
            role=MessageRoles.TOOL,
            content=content,
            name=tool_name,
            tool_call_id=tool_call_id
        )
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRoles.SYSTEM, content=content)
    
    @classmethod
    def from_litellm_message(cls, litellm_msg: Any) -> "Message":
        """Convert a LiteLLM response message to our Message format.
        
        Args:
            litellm_msg: The message object from LiteLLM response.choices[0].message
            
        Returns:
            Message instance
        """
        # Check if message has tool calls
        if hasattr(litellm_msg, 'tool_calls') and litellm_msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in litellm_msg.tool_calls
            ]
            return cls.assistant_with_tools(tool_calls)
        else:
            # Regular assistant message with text content
            content = litellm_msg.content or ""
            return cls.assistant(content)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary (for deserialization from state)."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values for API compatibility."""
        return self.model_dump(exclude_none=True)
    
    def to_dict_with_none(self) -> Dict[str, Any]:
        """Convert to dictionary, including None values (for state storage)."""
        return self.model_dump()


# Type aliases for state structure
ContextDict: TypeAlias = Dict[str, Any]  # Serialized AgentContext
MessageDict: TypeAlias = Dict[str, Any]  # Serialized Message


class AgentState(TypedDict):
    """LangGraph state for the agentic research agent.
    
    Message History:
    - Stored as dicts in OpenAI-compatible format for LangGraph serialization
    - Use Message.from_dict() to load as Pydantic models
    - Use Message.to_dict() to convert back for state storage
    
    Context:
    - Stored as dict for LangGraph compatibility
    - Use AgentContext.from_dict() to load as Pydantic model
    """
    messages: Annotated[List[MessageDict], operator.add]  # Message history (as dicts)
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


class TokenUsage(BaseModel):
    """Token usage statistics from LLM response."""
    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")
    reasoning_tokens: Optional[int] = Field(default=None, description="Number of tokens used for reasoning (o1, GPT-4o with reasoning)")
    non_reasoning_tokens: Optional[int] = Field(default=None, description="Number of non-reasoning completion tokens")
    
    @classmethod
    def from_litellm_usage(cls, usage: Any) -> "TokenUsage":
        """Extract token usage from LiteLLM response usage object."""
        if not usage:
            return cls()
        
        token_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }
        
        # Check for reasoning tokens (GPT-4o with reasoning, o1, etc.)
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                token_data["reasoning_tokens"] = details.reasoning_tokens
                token_data["non_reasoning_tokens"] = token_data["completion_tokens"] - details.reasoning_tokens
        
        return cls(**token_data)


class LLMResponseMetadata(BaseModel):
    """Metadata from LLM response including tokens and reasoning content."""
    tokens: Optional[TokenUsage] = Field(default=None, description="Token usage statistics")
    reasoning_content: Optional[str] = Field(default=None, description="Extended thinking/reasoning content from models like o1")
    model: Optional[str] = Field(default=None, description="Model that generated the response")
    
    @classmethod
    def from_litellm_response(cls, response: Any) -> "LLMResponseMetadata":
        """Extract metadata from LiteLLM response object.
        
        Attempts to extract:
        - Token usage (including reasoning tokens)
        - Reasoning content from various possible fields
        - Model name
        """
        metadata = cls()
        
        # Extract token usage
        if hasattr(response, 'usage') and response.usage:
            metadata.tokens = TokenUsage.from_litellm_usage(response.usage)
        
        # Extract model name
        if hasattr(response, 'model'):
            metadata.model = response.model
        
        # Extract reasoning content - try multiple possible locations
        reasoning_content = None
        response_message = None
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                response_message = choice.message
                
                # Try message-level fields
                if hasattr(response_message, 'reasoning_content') and response_message.reasoning_content:
                    reasoning_content = response_message.reasoning_content
                elif hasattr(response_message, 'reasoning') and response_message.reasoning:
                    reasoning_content = response_message.reasoning
            
            # Try choice-level fields
            if not reasoning_content:
                if hasattr(choice, 'reasoning_content') and choice.reasoning_content:
                    reasoning_content = choice.reasoning_content
                elif hasattr(choice, 'reasoning') and choice.reasoning:
                    reasoning_content = choice.reasoning
        
        # Try response-level fields
        if not reasoning_content:
            if hasattr(response, 'reasoning_content') and response.reasoning_content:
                reasoning_content = response.reasoning_content
            elif hasattr(response, 'reasoning') and response.reasoning:
                reasoning_content = response.reasoning
        
        if reasoning_content:
            metadata.reasoning_content = reasoning_content
        
        return metadata
    
    def to_dict(self) -> dict:
        """Convert to dict for logging, excluding None values."""
        data = {}
        
        if self.tokens:
            data["tokens"] = self.tokens.model_dump(exclude_none=True)
        
        if self.reasoning_content:
            data["reasoning_content"] = self.reasoning_content
        
        if self.model:
            data["model"] = self.model
        
        return data


# ============================================================================
# SSE Event Models for Streaming
# ============================================================================

class EventTypes:
    """Constants for SSE event types."""
    AGENT_REASONING = "agent_reasoning"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    CHECKLIST_UPDATED = "checklist_updated"
    SOURCE_DISCOVERED = "source_discovered"
    FINAL_REPORT = "final_report"
    ERROR = "error"
    COMPLETE = "complete"


class StreamEvent(BaseModel):
    """Base model for streaming events."""
    event_type: str = Field(description="Type of event")
    data: Dict[str, Any] = Field(description="Event payload")
    
    def to_sse(self) -> str:
        """Convert to SSE format: event: type\ndata: json\n\n"""
        import json
        return f"event: {self.event_type}\ndata: {json.dumps(self.data)}\n\n"


class AgentReasoningEvent(BaseModel):
    """Event emitted when agent produces reasoning text."""
    content: str
    
    def to_stream_event(self) -> StreamEvent:
        return StreamEvent(
            event_type=EventTypes.AGENT_REASONING,
            data={
                "content": self.content
            }
        )


class ToolCallStartedEvent(BaseModel):
    """Event emitted when a tool call starts."""
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any]
    
    def to_stream_event(self) -> StreamEvent:
        return StreamEvent(
            event_type=EventTypes.TOOL_CALL_STARTED,
            data={
                "tool_name": self.tool_name,
                "tool_call_id": self.tool_call_id,
                "arguments": self.arguments
            }
        )


class ToolCallCompletedEvent(BaseModel):
    """Event emitted when a tool call completes."""
    tool_name: str
    tool_call_id: str
    success: bool
    result_preview: Optional[str] = None  # Short preview of result
    
    def to_stream_event(self) -> StreamEvent:
        return StreamEvent(
            event_type=EventTypes.TOOL_CALL_COMPLETED,
            data={
                "tool_name": self.tool_name,
                "tool_call_id": self.tool_call_id,
                "success": self.success,
                "result_preview": self.result_preview
            }
        )


class ChecklistUpdatedEvent(BaseModel):
    """Event emitted when checklist is modified."""
    items: Dict[str, Dict[str, Any]]  # Full checklist state
    action: str  # "added", "completed", "updated"
    affected_item_ids: List[str]
    
    def to_stream_event(self) -> StreamEvent:
        return StreamEvent(
            event_type=EventTypes.CHECKLIST_UPDATED,
            data={
                "items": self.items,
                "action": self.action,
                "affected_item_ids": self.affected_item_ids
            }
        )


class SourceDiscoveredEvent(BaseModel):
    """Event emitted when new sources are found."""
    sources: List[Dict[str, Any]]  # List of Source dicts
    
    def to_stream_event(self) -> StreamEvent:
        return StreamEvent(
            event_type=EventTypes.SOURCE_DISCOVERED,
            data={
                "sources": self.sources
            }
        )


class FinalReportEvent(BaseModel):
    """Event emitted when final report is generated."""
    report: str
    
    def to_stream_event(self) -> StreamEvent:
        return StreamEvent(
            event_type=EventTypes.FINAL_REPORT,
            data={
                "report": self.report
            }
        )



