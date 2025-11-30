"""Handlers for tool execution with context injection.

Pattern: All tools receive context - it's always injected unconditionally.
"""
import json
import logging
from typing import Tuple, Dict, Any, List, Generator, Optional
from models import (
    AgentContext, StateKeys, ToolNames, Message,
    ToolCallStartedEvent, ToolCallCompletedEvent,
    ChecklistUpdatedEvent, SourceDiscoveredEvent,
    StreamEvent
)

logger = logging.getLogger(__name__)


def execute_tool_with_context(
    tool_func, 
    tool_name: str, 
    arguments: Dict[str, Any], 
    context: AgentContext
) -> Tuple[str, Dict[str, Any]]:
    """Execute a single tool with context always injected.
    
    All tools accept a 'context' parameter. It's injected here regardless of
    whether the tool uses it or not.
    
    Args:
        tool_func: The tool function
        tool_name: Tool name
        arguments: Tool arguments from LLM
        context: Shared context (always injected)
    
    Returns:
        Tuple of (display_message, state_updates_dict)
    """
    try:
        logger.info(f"🔧 {tool_name}")
        
        # Always inject context into arguments
        arguments['context'] = context
        
        # Execute tool
        result = tool_func.invoke(arguments)
        
        # Handle state updates
        state_updates: Dict[str, Any] = {}
        if tool_name == ToolNames.WRITE_FINAL_REPORT:
            state_updates[StateKeys.FINAL_REPORT] = result.final_report
            logger.info(f"  → 📊 Final report set: {len(result.final_report)} chars")
        
        logger.info(f"✓ {tool_name}")
        return str(result), state_updates
    
    except Exception as e:
        logger.error(f"❌ {tool_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Tool error: {str(e)}", {}


def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    available_tools: List,
    context: AgentContext,
    event_callback: Optional[callable] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute multiple tool calls and return tool messages and state updates.
    
    Args:
        tool_calls: List of tool call dicts from assistant message
        available_tools: List of available tool functions
        context: Agent context (will be modified in place)
        event_callback: Optional callback to send streaming events
        
    Returns:
        Tuple of (tool_messages, state_updates)
    """
    logger.info(f"🔧 Executing {len(tool_calls)} tool(s)...")
    
    tool_messages: List[Dict[str, Any]] = []
    state_updates: Dict[str, Any] = {}
    
    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        
        # Parse arguments with error handling
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in tool arguments for {function_name}: {e}")
            tool_msg = Message.tool(
                content=f"Error: Invalid JSON in arguments - {str(e)}",
                tool_name=function_name,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_msg.to_dict_with_none())
            continue
        
        # Find the tool function
        tool_func = next((t for t in available_tools if t.name == function_name), None)
        if not tool_func:
            logger.warning(f"⚠️  Unknown tool: {function_name}")
            tool_msg = Message.tool(
                content=f"Error: Unknown tool '{function_name}'",
                tool_name=function_name,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_msg.to_dict_with_none())
            continue
        
        # Execute tool with context
        logger.info(f"  → Executing {function_name} with args: {list(arguments.keys())}")
        
        # Emit tool started event
        if event_callback:
            started_event = ToolCallStartedEvent(
                tool_name=function_name,
                tool_call_id=tool_call["id"],
                arguments={k: v for k, v in arguments.items() if k != 'context'}  # Don't send context
            )
            event_callback(started_event.to_stream_event())
        
        try:
            # Track context state before tool execution
            sources_before = len(context.sources)
            checklist_items_before = set(context.checklist.items.keys())
            
            display_message, additional_updates = execute_tool_with_context(
                tool_func, function_name, arguments, context
            )
            logger.info(f"  ✓ {function_name} completed")
            
            # Detect and emit events for context changes
            if event_callback:
                # Check for new sources
                sources_after = len(context.sources)
                if sources_after > sources_before:
                    new_sources = context.sources[sources_before:]
                    source_event = SourceDiscoveredEvent(
                        sources=[src.model_dump() for src in new_sources]
                    )
                    event_callback(source_event.to_stream_event())
                
                # Check for checklist changes
                checklist_items_after = set(context.checklist.items.keys())
                if checklist_items_after != checklist_items_before:
                    new_items = checklist_items_after - checklist_items_before
                    if new_items:
                        action = "added"
                        affected_ids = list(new_items)
                    else:
                        # Check for status changes
                        action = "updated"
                        affected_ids = list(checklist_items_before)
                    
                    checklist_event = ChecklistUpdatedEvent(
                        items=context.checklist.to_dict(),
                        action=action,
                        affected_item_ids=affected_ids
                    )
                    event_callback(checklist_event.to_stream_event())
            
            # Emit tool completed event
            if event_callback:
                # Create a short preview (first 100 chars)
                preview = display_message[:100] + "..." if len(display_message) > 100 else display_message
                completed_event = ToolCallCompletedEvent(
                    tool_name=function_name,
                    tool_call_id=tool_call["id"],
                    success=True,
                    result_preview=preview
                )
                event_callback(completed_event.to_stream_event())
            
            # Add tool result message
            tool_msg = Message.tool(
                content=display_message,
                tool_name=function_name,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_msg.to_dict_with_none())
            
            # Collect state updates (like final_report)
            state_updates.update(additional_updates)
            
        except Exception as e:
            logger.error(f"❌ Error executing {function_name}: {e}")
            
            # Emit error event
            if event_callback:
                error_event = ToolCallCompletedEvent(
                    tool_name=function_name,
                    tool_call_id=tool_call["id"],
                    success=False,
                    result_preview=f"Error: {str(e)}"
                )
                event_callback(error_event.to_stream_event())
            
            tool_msg = Message.tool(
                content=f"Error executing tool: {str(e)}",
                tool_name=function_name,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_msg.to_dict_with_none())
    
    # Save updated context back to state
    state_updates[StateKeys.CONTEXT] = context.to_dict()
    
    return tool_messages, state_updates

