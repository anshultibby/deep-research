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
import logging
from typing import Literal, Dict, List, Any, Generator, Optional
from langgraph.graph import StateGraph, END
from litellm import completion, AuthenticationError, RateLimitError, APIError
import config
from models import (
    AgentState, 
    AgentContext, 
    Message, 
    LLMResponseMetadata,
    StateKeys,
    MessageRoles,
    ToolNames,
    StreamEvent,
    FinalReportEvent
)
from prompts import AGENT_SYSTEM_PROMPT
from tools import get_all_tools
from tool_handlers import execute_tool_calls
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
        
        # Event callback for streaming (set during streaming mode)
        self._event_callback: Optional[callable] = None
    
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
    
    def _agent_node(self, state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
        """Agent node that decides what to do next."""
        message_dicts = state[StateKeys.MESSAGES]
        logger.info(f"🧠 Agent thinking... ({len(message_dicts)} messages)")
        
        try:
            # Prepare system message
            system_message = Message.system(AGENT_SYSTEM_PROMPT)
            
            # Remove None values from messages (API compatibility)
            clean_messages = [
                {k: v for k, v in msg.items() if v is not None}
                for msg in message_dicts
            ]
            
            # Call LLM with tools
            logger.info(f"🤖 Calling {self.model} with {len(self.tool_schemas)} tools...")
            
            llm_params = {
                "model": self.model,
                "messages": [system_message.to_dict()] + clean_messages,
                "tools": self.tool_schemas,
                "tool_choice": "auto"
            }
            
            response = completion(**llm_params)

        except AuthenticationError as e:
            logger.error(f"❌ Authentication error: {e}")
            raise ValueError(f"Invalid API key for model {self.model}") from e
        except RateLimitError as e:
            logger.error(f"❌ Rate limit error: {e}")
            raise ValueError(f"Rate limit exceeded for model {self.model}") from e
        except APIError as e:
            logger.error(f"❌ API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in agent node: {e}")
            raise
        
        # Convert LiteLLM response to our Message format
        litellm_message = response.choices[0].message
        new_message = Message.from_litellm_message(litellm_message)
        
        # Log assistant message with metadata
        chat_logger = get_chat_logger()
        llm_metadata = LLMResponseMetadata.from_litellm_response(response)
        metadata_dict = llm_metadata.to_dict()
        
        # Add metadata to message for logging
        message_with_metadata = {**new_message.to_dict_with_none(), "metadata": metadata_dict} if metadata_dict else new_message.to_dict_with_none()
        chat_logger.log_message(message_with_metadata)
        
        # Return message as dict for state storage
        return {StateKeys.MESSAGES: [new_message.to_dict_with_none()]}
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on agent's decision."""
        last_message = state[StateKeys.MESSAGES][-1]
        
        # Get tool calls from last assistant message
        tool_calls = last_message.get("tool_calls", [])
        if not tool_calls:
            logger.warning("⚠️  No tool calls found in message")
            return {StateKeys.MESSAGES: []}
        
        try:
            # Load context (shared across all tool calls)
            context = AgentContext.from_dict(state[StateKeys.CONTEXT])
            
            # Execute all tool calls (with optional event streaming)
            tool_messages, state_updates = execute_tool_calls(
                tool_calls=tool_calls,
                available_tools=self.tools,
                context=context,
                event_callback=self._event_callback  # Pass callback for streaming
            )
            
            # Log tool messages
            chat_logger = get_chat_logger()
            chat_logger.log_messages(tool_messages)
            
            # Emit final report event if it was generated
            if StateKeys.FINAL_REPORT in state_updates and state_updates[StateKeys.FINAL_REPORT]:
                if self._event_callback:
                    report_event = FinalReportEvent(report=state_updates[StateKeys.FINAL_REPORT])
                    self._event_callback(report_event.to_stream_event())
            
            return {
                StateKeys.MESSAGES: tool_messages,
                **state_updates
            }
            
        except Exception as e:
            logger.error(f"❌ Error in tool node: {e}")
            raise
    
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide if agent should continue or end."""
        last_message = state[StateKeys.MESSAGES][-1]
        
        # If assistant wants to call tools, continue to tools node
        if last_message.get("tool_calls"):
            return "continue"
        
        # Check if final report exists
        if state.get(StateKeys.FINAL_REPORT):
            logger.info("✅ Final report exists - research complete")
            return "end"
        
        # Safety: max iterations
        message_count = len(state[StateKeys.MESSAGES])
        if message_count > self.max_iterations * 2:
            logger.warning(f"⚠️  Max iterations reached ({message_count} messages) - forcing end")
            return "end"
        
        # Check if checklist is complete but agent hasn't finished
        context = AgentContext.from_dict(state[StateKeys.CONTEXT])
        total_items = len(context.checklist.items)
        completed_items = sum(
            1 for item in context.checklist.items.values() 
            if item.status == "completed"
        )
        
        if total_items > 0 and completed_items == total_items and message_count > 10:
            logger.warning(
                f"⚠️  All {total_items} items complete but no finish call - "
                f"agent may be looping"
            )
        
        return "continue"
    
    def research(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the research agent.
        
        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}]
        
        Returns:
            Dictionary with messages, context, and final_report (AgentState)
        """
        # Start new chat session for logging
        chat_logger = get_chat_logger()
        session_id = chat_logger.start_session()
        logger.info(f"📝 Chat session: {session_id}")
        
        # Initialize with empty context
        context = AgentContext()
        
        # Convert incoming messages to Message objects, then to dicts for state
        formatted_messages: List[Dict[str, Any]] = []
        for m in messages:
            msg = Message.from_dict(m)
            formatted_messages.append(msg.to_dict_with_none())
        
        # Log initial messages
        chat_logger.log_messages(formatted_messages)
        
        # Create initial state
        initial_state: AgentState = {
            StateKeys.MESSAGES: formatted_messages,
            StateKeys.CONTEXT: context.to_dict(),
            StateKeys.FINAL_REPORT: None
        }
        
        # Run the agent graph
        result = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 50}  # Allow more iterations for thorough research
        )
        
        # Log final result
        logger.info(f"🔍 Result keys: {list(result.keys())}")
        final_report = result.get(StateKeys.FINAL_REPORT)
        logger.info(f"📊 final_report present: {bool(final_report)}")
        
        if final_report:
            logger.info(f"✅ Final report length: {len(final_report)} chars")
            completion_msg = Message.system(
                f"Final report generated ({len(final_report)} chars)"
            )
            completion_dict = completion_msg.to_dict_with_none()
            completion_dict["type"] = "completion"
            chat_logger.log_message(completion_dict)
        else:
            logger.warning("⚠️  No final_report in result!")
        
        logger.info(f"📝 Chat log: {chat_logger.get_session_file()}")
        
        return result
    
    def research_stream(
        self, 
        messages: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Run the research agent with streaming events via SSE.
        
        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}]
        
        Yields:
            SSE-formatted event strings
        """
        # Start new chat session for logging
        chat_logger = get_chat_logger()
        session_id = chat_logger.start_session()
        logger.info(f"📝 Chat session (streaming): {session_id}")
        
        # Initialize with empty context
        context = AgentContext()
        
        # Convert incoming messages to Message objects
        formatted_messages: List[Dict[str, Any]] = []
        for m in messages:
            msg = Message.from_dict(m)
            formatted_messages.append(msg.to_dict_with_none())
        
        # Log initial messages
        chat_logger.log_messages(formatted_messages)
        
        # Create initial state
        initial_state: AgentState = {
            StateKeys.MESSAGES: formatted_messages,
            StateKeys.CONTEXT: context.to_dict(),
            StateKeys.FINAL_REPORT: None
        }
        
        # Event queue for streaming
        events_to_send: List[StreamEvent] = []
        
        def event_callback(event: StreamEvent):
            """Callback to collect events during graph execution."""
            events_to_send.append(event)
        
        # Set callback for this streaming session
        self._event_callback = event_callback
        
        try:
            # Run the agent graph (using invoke, but with event callback)
            # Note: LangGraph's stream() doesn't work well with our tool pattern,
            # so we use invoke with callbacks instead
            result = self.graph.invoke(
                initial_state,
                config={"recursion_limit": 50}
            )
            
            # Send all collected events
            for event in events_to_send:
                yield event.to_sse()
            
            # Send completion event
            completion_event = StreamEvent(
                event_type="complete",
                data={
                    "message": "Research complete",
                    "context": result.get(StateKeys.CONTEXT),
                    "messages": result.get(StateKeys.MESSAGES)
                }
            )
            yield completion_event.to_sse()
            
            logger.info(f"📝 Chat log: {chat_logger.get_session_file()}")
            
        except Exception as e:
            logger.error(f"❌ Error in streaming research: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Send error event
            error_event = StreamEvent(
                event_type="error",
                data={"error": str(e)}
            )
            yield error_event.to_sse()
        
        finally:
            # Clear callback
            self._event_callback = None


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
