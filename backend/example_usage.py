"""Example usage of the agentic deep research agent."""
from research_agent import DeepResearchAgent


def example_basic_research():
    """Example of autonomous research."""
    agent = DeepResearchAgent(model="gpt-5", max_iterations=15)
    
    query = "What are the main breakthroughs in quantum computing hardware in 2024?"
    messages = [{"role": "user", "content": query}]
    
    print(f"🔍 Query: {query}\n")
    print("=" * 100)
    
    result = agent.research(messages)
    
    # Display the agent's workflow
    print("\n🤖 Agent Actions:")
    for msg in result["messages"]:
        if msg["role"] == "tool":
            print(f"  🔧 {msg['name']}: {msg['content'][:100]}...")
    
    # Display checklist
    print("\n✅ Research Checklist:")
    for item_id, item in result["checklist"].items():
        status_icon = "✓" if item["status"] == "completed" else "☐"
        print(f"  {status_icon} {item['question']}")
        if item["findings"]:
            print(f"     Sources: {', '.join(item['source_ids'])}")
    
    # Display sources
    print(f"\n📚 Total Sources: {len(result['sources'])}")
    for src in result["sources"][:5]:
        print(f"  [{src['id']}] {src['title']}")
        print(f"     {src['url']}")
    
    # Display final report
    if result.get("final_report"):
        print("\n" + "=" * 100)
        print("📊 FINAL REPORT:")
        print("=" * 100)
        print(result["final_report"])
        print("=" * 100)


def example_with_clarification():
    """Example where agent asks for clarification."""
    agent = DeepResearchAgent(model="gpt-5")
    
    # Vague query that should trigger clarification
    query = "Tell me about AI"
    messages = [{"role": "user", "content": query}]
    
    print(f"🔍 Initial Query: {query}\n")
    print("=" * 100)
    
    result = agent.research(messages)
    
    # Check if agent asked for clarification
    asked_clarification = False
    for msg in result["messages"]:
        if msg["role"] == "tool" and msg["name"] == "ask_clarification":
            print("\n❓ Agent asked for clarification:")
            print(msg["content"])
            asked_clarification = True
    
    if asked_clarification:
        print("\n💡 In a real scenario, you would:")
        print("1. Get user's answers to the clarification questions")
        print("2. Add them as a new user message")
        print("3. Continue the research with updated context")


def example_complex_research():
    """Example of complex multi-faceted research."""
    agent = DeepResearchAgent(model="gpt-5", max_iterations=20)
    
    query = "Compare the capabilities, pricing, and use cases of GPT-4, Claude 3, and Gemini Pro for enterprise applications"
    messages = [{"role": "user", "content": query}]
    
    print(f"🔍 Complex Query: {query}\n")
    print("=" * 100)
    
    result = agent.research(messages)
    
    # Count tool calls
    tool_counts = {}
    for msg in result["messages"]:
        if msg["role"] == "tool":
            tool_name = msg["name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    
    # Load context
    from models import AgentContext
    context = AgentContext.from_dict(result["context"])
    
    print("\n📊 Agent Activity:")
    print(f"  Total messages: {len(result['messages'])}")
    print(f"  Tool calls:")
    for tool, count in tool_counts.items():
        print(f"    - {tool}: {count}")
    
    print(f"\n  Checklist items: {len(context.checklist.items)}")
    print(f"  Sources gathered: {context.get_sources_count()}")
    
    # Show the research structure
    print("\n🗂️  Research Structure:")
    for item in context.checklist.items.values():
        status = "✓" if item.status == "completed" else "⧗"
        print(f"  {status} {item.question}")
        if item.findings:
            print(f"     └─ {len(item.source_ids)} sources used")
    
    # Display final report
    if result.get("final_report"):
        print("\n" + "=" * 100)
        print("📊 FINAL REPORT:")
        print("=" * 100)
        print(result["final_report"])
        print("=" * 100)


if __name__ == "__main__":
    print("EXAMPLE 1: Basic Autonomous Research")
    print("=" * 100)
    example_basic_research()
    
    print("\n\n")
    print("EXAMPLE 2: Research with Clarification")
    print("=" * 100)
    example_with_clarification()
    
    print("\n\n")
    print("EXAMPLE 3: Complex Multi-Faceted Research")
    print("=" * 100)
    example_complex_research()
