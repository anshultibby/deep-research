"""End-to-end test to verify tools work with context injection."""
import sys
sys.path.insert(0, '..')

from models import AgentContext, Checklist
from tools import get_all_tools
import json


def test_all_tools():
    """Test that all tools can be called with context."""
    print("\n" + "="*70)
    print("END-TO-END TEST: All Tools with Context")
    print("="*70)
    
    # Create test context
    context = AgentContext(
        checklist=Checklist(),
        sources=[],
        source_counter=0
    )
    
    # Get all tools
    tools = get_all_tools()
    
    print(f"\nFound {len(tools)} tools to test")
    
    results = []
    
    # Test each tool
    for tool in tools:
        tool_name = tool.name
        print(f"\n{'-'*70}")
        print(f"Testing: {tool_name}")
        print(f"{'-'*70}")
        
        # Check schema
        schema = tool.get_input_schema().model_json_schema()
        properties = list(schema.get('properties', {}).keys())
        required = schema.get('required', [])
        
        print(f"Schema properties: {properties}")
        print(f"Schema required: {required}")
        
        # Check if context is in schema (should NOT be)
        if 'context' in properties:
            print(f"❌ FAIL: context is in schema!")
            results.append((tool_name, False, "context in schema"))
            continue
        
        if 'kwargs' in properties:
            print(f"✅ PASS: kwargs in schema (context hidden)")
        else:
            print(f"✅ PASS: context not in schema")
        
        # Try calling the tool with context
        try:
            if tool_name == "get_current_checklist":
                result = tool.func(context=context)
            elif tool_name == "modify_checklist":
                result = tool.func(items=["Test item"], context=context)
            elif tool_name == "ask_clarification":
                result = tool.func(questions=["Test question?"], context=context)
            elif tool_name == "write_subreport":
                # First add an item to complete
                context.checklist.add_items(["Test"])
                result = tool.func(
                    item_id="item_1",
                    findings="Test findings",
                    source_urls=[],
                    context=context
                )
            elif tool_name == "write_final_report":
                result = tool.func(final_report="Test report", context=context)
            elif tool_name == "search":
                # Skip search as it requires API keys
                print("⏭️  SKIP: search requires API key")
                results.append((tool_name, True, "skipped (requires API)"))
                continue
            else:
                print(f"⚠️  WARN: Unknown tool {tool_name}")
                results.append((tool_name, False, "unknown tool"))
                continue
            
            print(f"✅ PASS: Tool executed successfully")
            print(f"Result type: {type(result).__name__}")
            results.append((tool_name, True, "success"))
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results.append((tool_name, False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for tool_name, success, msg in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {tool_name:30} {msg}")
    
    print("="*70)
    print(f"Result: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = test_all_tools()
    exit(0 if success else 1)

