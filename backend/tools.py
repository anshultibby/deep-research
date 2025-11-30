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
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import config  # Load environment variables
from models import AgentContext


def tool_with_context(func: Callable) -> Callable:
    """Decorator for tools that need context injection.
    
    Wraps a function so:
    - It can accept 'context: AgentContext' parameter
    - Context is excluded from OpenAI schema
    - Actual execution receives context from tool_handlers
    
    Usage:
        @tool_with_context
        def my_tool(arg1: str, context: AgentContext) -> Result:
            # Use context here
            context.add_sources(...)
            return Result(...)
    """
    # Wrapper just calls the original function
    # Context will be injected by tool_handlers before calling
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Mark wrapper that it needs context injection
    wrapper._needs_context = True
    
    # Apply @tool decorator for LangChain
    return tool(wrapper)


# Tool return models with display methods
class SearchResult(BaseModel):
    """Result from search tool."""
    results: List[dict] = Field(description="List of search results with url, title, content")
    
    def __str__(self) -> str:
        if not self.results:
            return "<search_result>\n<status>No results found</status>\n</search_result>"
        
        if "error" in self.results[0]:
            return f"<search_result>\n<error>{self.results[0]['error']}</error>\n</search_result>"
        
        output = f"<search_result>\n<summary>Found {len(self.results)} sources</summary>\n\n"
        for idx, r in enumerate(self.results, 1):
            content = r.get('content', 'No content')
            # Show up to 5,000 chars of content for the agent to read
            truncated_content = content[:5000] if len(content) > 5000 else content
            
            output += f"<source id='{idx}'>\n"
            output += f"<title>{r.get('title', 'No title')}</title>\n"
            output += f"<url>{r.get('url', 'No URL')}</url>\n"
            output += f"<content>\n{truncated_content}\n</content>\n"
            output += f"</source>\n\n"
        
        output += "</search_result>"
        return output


class ClarificationRequest(BaseModel):
    """Request for user clarification."""
    action: str = Field(default="pause_for_user")
    questions: List[str] = Field(description="Clarifying questions")
    
    def __str__(self) -> str:
        output = "<clarification_request>\n<message>Asking user for clarification</message>\n\n<questions>\n"
        for i, q in enumerate(self.questions, 1):
            output += f"<question id='{i}'>{q}</question>\n"
        output += "</questions>\n\n<status>Execution paused - waiting for user input</status>\n</clarification_request>"
        return output


class ChecklistView(BaseModel):
    """View of current checklist."""
    checklist_display: str = Field(description="Formatted checklist with status")
    total_items: int = Field(description="Total number of checklist items")
    completed_items: int = Field(description="Number of completed items")
    sources_count: int = Field(description="Number of sources discovered")
    
    def __str__(self) -> str:
        output = "<checklist_view>\n"
        output += f"<summary>Total: {self.total_items} items | Completed: {self.completed_items} | Sources: {self.sources_count}</summary>\n\n"
        output += f"<items>\n{self.checklist_display}\n</items>\n"
        output += "</checklist_view>"
        return output


class ChecklistUpdate(BaseModel):
    """Update to checklist."""
    action: str = Field(default="update_checklist")
    items: List[str] = Field(description="Checklist items to add")
    
    def __str__(self) -> str:
        output = "<checklist_update>\n<message>Updated checklist with new items</message>\n\n<new_items>\n"
        for item in self.items:
            output += f"<item status='pending'>☐ {item}</item>\n"
        output += "</new_items>\n</checklist_update>"
        return output


class SubreportComplete(BaseModel):
    """Subreport completion."""
    item_id: str = Field(description="Checklist item ID")
    findings: str = Field(description="Research findings")
    source_urls: List[str] = Field(description="Source URLs used")
    
    def __str__(self) -> str:
        output = "<subreport_complete>\n"
        output += f"<item_id>{self.item_id}</item_id>\n"
        output += f"<status>✓ Completed</status>\n"
        output += f"<findings>\n{self.findings}\n</findings>\n"
        output += f"<sources count='{len(self.source_urls)}'>\n"
        for url in self.source_urls:
            output += f"<url>{url}</url>\n"
        output += "</sources>\n</subreport_complete>"
        return output


class ResearchComplete(BaseModel):
    """Research completion with final report."""
    final_report: str = Field(description="The complete research report")
    
    def __str__(self) -> str:
        output = "<research_complete>\n"
        output += "<status>✓ Research completed! Final report written</status>\n"
        output += f"<report length='{len(self.final_report)}'>\n{self.final_report}\n</report>\n"
        output += "</research_complete>"
        return output


# ============================================================================
# Helper Functions
# ============================================================================

def scrape_page_content(url: str, timeout: int = 5) -> str:
    """Scrape full text content from a webpage.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        Scraped text content (up to 5,000 chars)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Truncate to 5k chars for reasonable size
        return text[:5000] if len(text) > 5000 else text
        
    except Exception as e:
        return f"[Failed to scrape: {str(e)}]"


# ============================================================================
# Tool Definitions
# ============================================================================

@tool_with_context
def search(query: Annotated[str, "The search query to look up"], context: AgentContext) -> SearchResult:
    """Search the web for information on a specific query. Returns sources with URLs, titles, and full scraped content."""
    # Get API key from config
    serper_api_key = config.SERPER_API_KEY
    if not serper_api_key:
        return SearchResult(results=[{"error": "SERPER_API_KEY not set"}])
    
    # Search using Serper
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 3}
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
    results = data.get("organic", [])[:3]
    
    if not results:
        return SearchResult(results=[])
    
    # Extract URLs and titles
    search_results = [
        {
            "url": r.get("link", ""),
            "title": r.get("title", ""),
            "snippet": r.get("snippet", "")
        }
        for r in results
    ]
    
    # Scrape full content from all URLs in parallel
    formatted_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all scraping tasks
        future_to_result = {
            executor.submit(scrape_page_content, r["url"]): r 
            for r in search_results
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_result):
            result = future_to_result[future]
            scraped_content = future.result()
            
            formatted_results.append({
                "url": result["url"],
                "title": result["title"],
                "content": scraped_content  # Full scraped content
            })
    
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
def write_final_report(
    final_report: Annotated[str, "The complete research report with citations"]
) -> ResearchComplete:
    """Write the final research report that synthesizes all findings with proper citations. This completes the research."""
    return ResearchComplete(final_report=final_report)


# Helper to get all tools as a list
def get_all_tools():
    """Return all tools for the agent."""
    return [search, ask_clarification, get_current_checklist, modify_checklist, write_subreport, write_final_report]
