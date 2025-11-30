"""Handlers for tool execution with context injection.

Pattern: Tools decorated with @tool_with_context automatically receive context.
"""
import logging
import inspect
from models import AgentContext

logger = logging.getLogger(__name__)


def execute_tool_with_context(tool_func, tool_name: str, arguments: dict, context: AgentContext) -> tuple[str, dict]:
    """Execute a tool with context injected if needed.
    
    Checks if tool needs context (via _needs_context marker) and injects it.
    
    Args:
        tool_func: The tool function
        tool_name: Tool name
        arguments: Tool arguments from LLM
        context: Shared context (injected if tool needs it)
    
    Returns:
        (display_message, state_updates_dict)
    """
    try:
        logger.info(f"🔧 {tool_name}")
        
        # Check if tool needs context injection
        needs_context = getattr(tool_func.func, '_needs_context', False) if hasattr(tool_func, 'func') else False
        
        if needs_context:
            # Inject context into arguments
            arguments['context'] = context
        
        # Execute tool
        result = tool_func.invoke(arguments)
        
        # Handle state updates
        state_updates = {}
        if tool_name == "write_final_report":
            state_updates["final_report"] = result.final_report
            logger.info(f"  → 📊 Final report set: {len(result.final_report)} chars")
        
        logger.info(f"✓ {tool_name}")
        return str(result), state_updates
    
    except Exception as e:
        logger.error(f"❌ {tool_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Tool error: {str(e)}", {}

