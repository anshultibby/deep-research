"""Agentic deep research agent using LangGraph and LiteLLM.

Message History Management:
---------------------------
Messages are stored in OpenAI-compatible format for simplicity and clarity:

1. User message:
   {"role": "user", "content": "What is quantum computing?"}

2. Assistant text response:
   {"role": "assistant", "content": "Quantum computing is..."}

3. Assistant calling tools:
   {"role": "assistant", "content": None, "tool_calls": [
       {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{...}"}}
   ]}

4. Tool result:
   {"role": "tool", "content": "Search results...", "tool_call_id": "call_123", "name": "search"}

This format:
- Is compatible with OpenAI API directly (no conversion needed)
- Preserves full tool call structure
- Makes debugging easy (you can see exactly what's sent to LLM)
- Works with any LLM provider through LiteLLM
"""
import os
import json
import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from litellm import completion
import config  # Load environment variables
from models import AgentState, AgentContext, Message, LLMResponseMetadata
from prompts import AGENT_SYSTEM_PROMPT
from tools import get_all_tools
from tool_handlers import execute_tool_with_context
from chat_logger import get_chat_logger

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """An agentic deep research agent that autonomously researches topics."""
    
    def __init__(
        self,
        model: str = "gpt-5",
        max_iterations: int = 15
    ):
        self.model = model
        self.max_iterations = max_iterations
        
        # Get tools from tools.py
        self.tools = get_all_tools()
        
        # Convert to LiteLLM format (tools have .name, .description, .args_schema)
        self.tool_schemas = self._convert_tools_to_schemas()
        
        self.graph = self._build_graph()
    
    def _convert_tools_to_schemas(self) -> list:
        """Convert LangChain tools to LiteLLM tool schemas."""
        schemas = []
        for tool in self.tools:
            # LangChain tools have name, description, and args_schema
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {}
                }
            }
            schemas.append(schema)
        return schemas
    
    def _build_graph(self) -> StateGraph:
        """Build the agentic LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Single agent node that decides what to do
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        # Start with agent
        workflow.set_entry_point("agent")
        
        # Agent decides: continue with tools or end
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # After tools, go back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> dict:
        """Agent node that decides what to do next."""
        try:
            messages = state["messages"]
            logger.info(f"🧠 Agent thinking... ({len(messages)} messages)")
            
            # Build system prompt (no dynamic content - agent uses get_current_checklist tool)
            system_message = {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT
            }
            
            # Messages are already in OpenAI-compatible format - just pass them through
            # Filter out None values from messages for clean API calls
            clean_messages = []
            for msg in messages:
                clean_msg = {k: v for k, v in msg.items() if v is not None}
                clean_messages.append(clean_msg)
            
            # Call LLM with tools
            logger.info(f"🤖 Calling {self.model} with {len(self.tool_schemas)} tools...")
            
            # GPT-5 only supports temperature=1, other models support configurable temperature
            llm_params = {
                "model": self.model,
                "messages": [system_message] + clean_messages,
                "tools": self.tool_schemas,
                "tool_choice": "auto"
            }
            
            # Only add temperature for non-GPT-5 models
            if not self.model.startswith("gpt-5"):
                llm_params["temperature"] = 0.7
            
            response = completion(**llm_params)
            
            # Log token usage
            usage = response.usage if hasattr(response, 'usage') else None
            if usage:
                logger.info(f"✅ LLM response: {usage.prompt_tokens} in, {usage.completion_tokens} out")
            else:
                logger.info("✅ LLM response received")
        except Exception as e:
            logger.error(f"❌ Error in agent node: {e}")
            raise
        
        response_message = response.choices[0].message
        
        # Store message in OpenAI-compatible format
        new_message: Message = {
            "role": "assistant",
            "content": response_message.content,
            "name": None,
            "tool_calls": None,
            "tool_call_id": None
        }
        
        # If LLM wants to call tools, store tool_calls
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            new_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response_message.tool_calls
            ]
            new_message["content"] = None  # No text content when calling tools
        
        # Log assistant message with metadata
        chat_logger = get_chat_logger()
        
        # Extract metadata using Pydantic model
        llm_metadata = LLMResponseMetadata.from_litellm_response(response)
        
        # Log info about captured data
        if llm_metadata.tokens:
            logger.info(f"✅ Token usage: {llm_metadata.tokens.prompt_tokens} in, {llm_metadata.tokens.completion_tokens} out")
            if llm_metadata.tokens.reasoning_tokens:
                logger.info(f"🧠 Reasoning tokens: {llm_metadata.tokens.reasoning_tokens}")
        
        if llm_metadata.reasoning_content:
            logger.info(f"🧠 Reasoning captured: {len(llm_metadata.reasoning_content)} chars")
        
        # Debug logging
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(f"Response structure: {dir(response)}")
            logger.debug(f"Message structure: {dir(response_message)}")
            if hasattr(response_message, '__dict__'):
                logger.debug(f"Message dict keys: {response_message.__dict__.keys()}")
        
        # If we have reasoning tokens but no reasoning content, log a warning
        if llm_metadata.tokens and llm_metadata.tokens.reasoning_tokens and llm_metadata.tokens.reasoning_tokens > 0:
            if not llm_metadata.reasoning_content:
                logger.warning(f"⚠️  Model used {llm_metadata.tokens.reasoning_tokens} reasoning tokens but no reasoning content was captured!")
                logger.warning(f"   Response message attrs: {[attr for attr in dir(response_message) if not attr.startswith('_')]}")
                # Try to get raw response data
                if hasattr(response, 'model_dump') or hasattr(response, 'dict'):
                    try:
                        raw_data = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                        logger.warning(f"   Raw response keys: {list(raw_data.keys())}")
                        if 'choices' in raw_data and len(raw_data['choices']) > 0:
                            logger.warning(f"   Choice[0] keys: {list(raw_data['choices'][0].keys())}")
                            if 'message' in raw_data['choices'][0]:
                                logger.warning(f"   Message keys: {list(raw_data['choices'][0]['message'].keys())}")
                    except Exception as e:
                        logger.warning(f"   Could not dump raw response: {e}")
        
        # Log with metadata (convert to dict for JSON serialization)
        metadata_dict = llm_metadata.to_dict()
        message_with_metadata = {**new_message, "metadata": metadata_dict} if metadata_dict else new_message
        chat_logger.log_message(message_with_metadata)
        
        return {"messages": [new_message]}
    
    def _tool_node(self, state: AgentState) -> dict:
        """Execute tools based on agent's decision."""
        try:
            last_message = state["messages"][-1]
            
            # Get tool calls from last assistant message
            tool_calls = last_message.get("tool_calls", [])
            if not tool_calls:
                logger.warning("⚠️  No tool calls found in message")
                return {"messages": []}
            
            logger.info(f"🔧 Executing {len(tool_calls)} tool(s)...")
            
            # Load context (shared across all tool calls)
            context = AgentContext.from_dict(state["context"])
            
            tool_messages = []
            state_updates = {}
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                # Find the tool
                tool_func = next((t for t in self.tools if t.name == function_name), None)
                if not tool_func:
                    tool_messages.append({
                        "role": "tool",
                        "content": f"Unknown tool: {function_name}",
                        "name": function_name,
                        "tool_call_id": tool_call["id"],
                        "tool_calls": None
                    })
                    continue
                
                # Execute tool with context
                logger.info(f"  → Executing {function_name} with args: {list(arguments.keys())}")
                display_message, additional_updates = execute_tool_with_context(
                    tool_func, function_name, arguments, context
                )
                logger.info(f"  ✓ {function_name} completed")
                
                # Add tool result message in OpenAI format
                tool_messages.append({
                    "role": "tool",
                    "content": display_message,
                    "name": function_name,
                    "tool_call_id": tool_call["id"],  # Required by OpenAI
                    "tool_calls": None  # Not applicable for tool messages
                })
                
                # Collect state updates (like final_report)
                state_updates.update(additional_updates)
        
            # Save updated context back to state
            state_updates["context"] = context.to_dict()
            
            # Log tool messages
            chat_logger = get_chat_logger()
            chat_logger.log_messages(tool_messages)
            
            return {
                "messages": tool_messages,
                **state_updates
            }
        except Exception as e:
            logger.error(f"❌ Error in tool node: {e}")
            raise
    
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide if agent should continue or end."""
        last_message = state["messages"][-1]
        
        # If assistant wants to call tools, continue to tools node
        if last_message.get("tool_calls"):
            return "continue"
        
        # Check if final report exists
        if state.get("final_report"):
            logger.info("✅ Final report exists - research complete")
            return "end"
        
        # Safety: max iterations
        if len(state["messages"]) > self.max_iterations * 2:
            logger.warning("⚠️  Max iterations reached - forcing end")
            return "end"
        
        # Check if checklist is complete but agent hasn't finished
        context = AgentContext.from_dict(state["context"])
        total_items = len(context.checklist.items)
        completed_items = sum(1 for item in context.checklist.items.values() if item.status == "completed")
        
        if total_items > 0 and completed_items == total_items and len(state["messages"]) > 10:
            logger.warning(f"⚠️  All {total_items} items complete but no finish call - agent may be looping")
        
        return "continue"
    
    def research(self, messages: list[dict]) -> dict:
        """Run the research agent.
        
        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}]
        
        Returns:
            Dictionary with messages, context, and final_report
        """
        # Start new chat session for logging
        chat_logger = get_chat_logger()
        session_id = chat_logger.start_session()
        logger.info(f"📝 Chat session: {session_id}")
        
        # Initialize with empty context
        context = AgentContext()
        
        # Format messages in OpenAI-compatible format
        formatted_messages = []
        for m in messages:
            msg = {
                "role": m["role"],
                "content": m.get("content"),
                "name": m.get("name"),
                "tool_calls": m.get("tool_calls"),
                "tool_call_id": m.get("tool_call_id")
            }
            formatted_messages.append(msg)
        
        # Log initial messages
        chat_logger.log_messages(formatted_messages)
        
        initial_state = {
            "messages": formatted_messages,
            "context": context.to_dict(),
            "final_report": None
        }
        
        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 50}  # Allow more iterations for thorough research
        )
        
        # Log final result
        logger.info(f"🔍 Result keys: {list(result.keys())}")
        logger.info(f"📊 final_report present: {bool(result.get('final_report'))}")
        
        if result.get("final_report"):
            logger.info(f"✅ Final report length: {len(result['final_report'])} chars")
            chat_logger.log_message({
                "role": "system",
                "content": f"Final report generated ({len(result['final_report'])} chars)",
                "type": "completion"
            })
        else:
            logger.warning("⚠️  No final_report in result!")
        
        logger.info(f"📝 Chat log: {chat_logger.get_session_file()}")
        
        return result


def main():
    """Example usage."""
    agent = DeepResearchAgent(model="gpt-5")
    
    query = "What are the main breakthroughs in quantum computing in 2024?"
    messages = [{"role": "user", "content": query}]
    
    print(f"🔍 Query: {query}\n")
    print("=" * 80)
    
    result = agent.research(messages)
    
    # Display final report
    if result.get("final_report"):
        print("\n📊 FINAL REPORT:")
        print("=" * 80)
        print(result["final_report"])
        print("=" * 80)
    
    # Load context for display
    context = AgentContext.from_dict(result["context"])
    
    # Display checklist
    print("\n✅ Research Checklist:")
    for item in context.checklist.items.values():
        status = "✓" if item.status == "completed" else "☐"
        print(f"  {status} {item.question}")
    
    # Display sources
    print(f"\n📚 Sources Used: {context.get_sources_count()}")
    for src in context.sources[:5]:
        print(f"  - {src.title} ({src.url})")


if __name__ == "__main__":
    main()
