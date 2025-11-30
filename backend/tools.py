"""Tool definitions for the agentic deep research agent.

Context Injection Pattern:
--------------------------
Tools are decorated with @tool_with_context which:
1. Allows tools to accept a 'context' parameter
2. Excludes 'context' from the LLM schema (OpenAI doesn't see it)
3. Context is injected at execution time by tool_handlers.py

This keeps tool definitions clean and natural.
"""
import os
import requests
import functools
from typing import List, Annotated, Callable, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import config  # Load environment variables
from models import AgentContext


def tool_with_context(func: Callable) -> Callable:
    """Decorator for tools that need context injection.
    
    Wraps a function so:
    - It can accept 'context: AgentContext' parameter
    - Context is excluded from OpenAI schema
    - Original @tool decorator still works for schema generation
    
    Usage:
        @tool_with_context
        def my_tool(arg1: str, context: AgentContext) -> Result:
            # Use context here
            context.add_sources(...)
            return Result(...)
    """
    # Mark that this function needs context injection
    func._needs_context = True
    
    # Create a wrapper for schema generation (without context parameter)
    @functools.wraps(func)
    def schema_wrapper(*args, **kwargs):
        # Remove context from kwargs for schema generation
        kwargs_no_context = {k: v for k, v in kwargs.items() if k != 'context'}
        # This is for schema generation - actual execution goes through tool_handlers
        return func(*args, **kwargs_no_context, context=None) if 'context' in func.__code__.co_varnames else func(*args, **kwargs_no_context)
    
    # Apply @tool decorator for LangChain
    return tool(schema_wrapper)


# Tool return models with display methods
class SearchResult(BaseModel):
    """Result from search tool."""
    results: List[dict] = Field(description="List of search results with url, title, content")
    
    def __str__(self) -> str:
        if not self.results:
            return "No results found."
        
        if "error" in self.results[0]:
            return self.results[0]["error"]
        
        output = f"Found {len(self.results)} sources:\n\n"
        for r in self.results:
            output += f"• {r.get('title', 'No title')}\n"
            output += f"  URL: {r.get('url', 'No URL')}\n"
            output += f"  {r.get('content', 'No content')[:100]}...\n\n"
        return output


class ClarificationRequest(BaseModel):
    """Request for user clarification."""
    action: str = Field(default="pause_for_user")
    questions: List[str] = Field(description="Clarifying questions")
    
    def __str__(self) -> str:
        output = "Asking user for clarification:\n\n"
        output += "\n".join([f"{i+1}. {q}" for i, q in enumerate(self.questions)])
        output += "\n\n(Execution paused - waiting for user input)"
        return output


class ChecklistView(BaseModel):
    """View of current checklist."""
    checklist_display: str = Field(description="Formatted checklist with status")
    total_items: int = Field(description="Total number of checklist items")
    completed_items: int = Field(description="Number of completed items")
    sources_count: int = Field(description="Number of sources discovered")
    
    def __str__(self) -> str:
        return self.checklist_display


class ChecklistUpdate(BaseModel):
    """Update to checklist."""
    action: str = Field(default="update_checklist")
    items: List[str] = Field(description="Checklist items to add")
    
    def __str__(self) -> str:
        output = "Updated checklist with new items:\n"
        output += "\n".join([f"☐ {item}" for item in self.items])
        return output


class SubreportComplete(BaseModel):
    """Subreport completion."""
    item_id: str = Field(description="Checklist item ID")
    findings: str = Field(description="Research findings")
    source_urls: List[str] = Field(description="Source URLs used")
    
    def __str__(self) -> str:
        return f"✓ Completed {self.item_id}: Documented findings with {len(self.source_urls)} sources"


class ResearchComplete(BaseModel):
    """Research completion with final report."""
    final_report: str = Field(description="The complete research report")
    
    def __str__(self) -> str:
        return "✓ Research completed! Final report written."


# ============================================================================
# Tool Definitions
# ============================================================================

@tool_with_context
def search(query: Annotated[str, "The search query to look up"], context: AgentContext) -> SearchResult:
    """Search the web for information on a specific query. Returns sources with URLs, titles, and content."""
    # Get API key from config
    serper_api_key = config.SERPER_API_KEY
    if not serper_api_key:
        return SearchResult(results=[{"error": "SERPER_API_KEY not set"}])
    
    # Search using Serper
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 5}
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return SearchResult(results=[{"error": f"Search failed: {str(e)}"}])
    
    # Get results
    results = data.get("organic", [])[:5]
    
    if not results:
        return SearchResult(results=[])
    
    # Format results
    formatted_results = [
        {
            "url": r.get("link", ""),
            "title": r.get("title", ""),
            "content": r.get("snippet", "")
        }
        for r in results
    ]
    
    # Update context with new sources
    context.add_sources(formatted_results)
    
    return SearchResult(results=formatted_results)


@tool
def ask_clarification(
    questions: Annotated[List[str], "List of 2-3 clarifying questions to ask the user"]
) -> ClarificationRequest:
    """Ask the user clarifying questions. Use ONLY at the beginning if the query is too vague. Pauses execution for user input."""
    return ClarificationRequest(questions=questions)


@tool_with_context
def get_current_checklist(context: AgentContext) -> ChecklistView:
    """Get the current state of your research checklist and progress. Shows what items are pending/completed and sources count."""
    return ChecklistView(
        checklist_display=context.checklist.format_display(),
        total_items=len(context.checklist.items),
        completed_items=sum(1 for i in context.checklist.items.values() if i.status == "completed"),
        sources_count=context.get_sources_count()
    )


@tool_with_context
def modify_checklist(
    items: Annotated[List[str], "List of research questions/topics to investigate"],
    context: AgentContext
) -> ChecklistUpdate:
    """Add or update research checklist items. Use to plan what needs to be researched."""
    # Update context with new items
    context.checklist.add_items(items)
    return ChecklistUpdate(items=items)


@tool_with_context
def write_subreport(
    item_id: Annotated[str, "The checklist item ID (e.g., 'item_1')"],
    findings: Annotated[str, "Your research findings (2-3 paragraphs)"],
    source_urls: Annotated[List[str], "URLs of sources used for these findings"],
    context: AgentContext
) -> SubreportComplete:
    """Write findings for a specific checklist item. Include the item_id, your findings, and source URLs used."""
    # Find source IDs and complete the checklist item
    source_ids = context.find_source_ids_by_urls(source_urls)
    context.checklist.complete_item(item_id, findings, source_ids)
    
    return SubreportComplete(item_id=item_id, findings=findings, source_urls=source_urls)


@tool
def finish(
    final_report: Annotated[str, "The complete research report with citations"]
) -> ResearchComplete:
    """Complete the research by writing a comprehensive final report that synthesizes all findings with proper citations."""
    return ResearchComplete(final_report=final_report)


# Helper to get all tools as a list
def get_all_tools():
    """Return all tools for the agent."""
    return [search, ask_clarification, get_current_checklist, modify_checklist, write_subreport, finish]
