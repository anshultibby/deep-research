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
from models import AgentState, AgentContext, Message
from prompts import AGENT_SYSTEM_PROMPT
from tools import get_all_tools
from tool_handlers import execute_tool_with_context

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
        
        # Check if agent called finish tool
        try:
            message_data = json.loads(last_message["content"])
            tool_calls = message_data.get("tool_calls", [])
            for tc in tool_calls:
                if tc["function"]["name"] == "finish":
                    return "end"
        except:
            pass
        
        # Check if final report exists
        if state.get("final_report"):
            return "end"
        
        # Safety: max iterations
        if len(state["messages"]) > self.max_iterations * 2:
            return "end"
        
        return "continue"
    
    def research(self, messages: list[dict]) -> dict:
        """Run the research agent.
        
        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}]
        
        Returns:
            Dictionary with messages, context, and final_report
        """
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
        
        initial_state = {
            "messages": formatted_messages,
            "context": context.to_dict(),
            "final_report": None
        }
        
        result = self.graph.invoke(initial_state)
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
