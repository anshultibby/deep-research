"""Unit test to verify context injection works with LangChain tools."""
from typing import Annotated
from functools import partial
from langchain_core.tools import tool
from models import AgentContext, Checklist


@tool
def test_tool_with_kwargs(query: Annotated[str, "A test query"], **kwargs) -> str:
    """A test tool that accepts context via kwargs."""
    context = kwargs.get('context')
    if context is None:
        return "ERROR: No context received"
    
    if not isinstance(context, AgentContext):
        return f"ERROR: Wrong type - got {type(context)}"
    
    return f"SUCCESS: Received context with {len(context.sources)} sources"


def test_method_1_invoke_with_kwargs():
    """Test 1: Use tool.invoke() with **kwargs approach."""
    print("\n=== Test 1: tool.invoke() with **kwargs ===\n")
    
    context = AgentContext(
        checklist=Checklist(),
        sources=[],
        source_counter=0
    )
    
    arguments = {"query": "test query", "context": context}
    result = test_tool_with_kwargs.invoke(arguments)
    
    print(f"Result: {result}")
    return result.startswith("SUCCESS")


def test_method_2_func_call_with_kwargs():
    """Test 2: Call tool.func() directly with **kwargs approach."""
    print("\n=== Test 2: tool.func() with **kwargs ===\n")
    
    context = AgentContext(
        checklist=Checklist(),
        sources=[],
        source_counter=0
    )
    
    # Call the underlying function directly
    result = test_tool_with_kwargs.func(query="test query", context=context)
    
    print(f"Result: {result}")
    return result.startswith("SUCCESS")


def test_method_3_check_schema():
    """Test 3: Check if context is in the schema."""
    print("\n=== Test 3: Schema Check ===\n")
    
    schema = test_tool_with_kwargs.get_input_schema().model_json_schema()
    properties = list(schema.get('properties', {}).keys())
    required = schema.get('required', [])
    
    print(f"Properties: {properties}")
    print(f"Required: {required}")
    
    context_in_schema = 'context' in properties
    if context_in_schema:
        print("❌ context is in schema - LLM will see it!")
        return False
    else:
        print("✅ context NOT in schema - LLM won't see it!")
        return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING CONTEXT INJECTION WITH **kwargs APPROACH")
    print("="*70)
    
    test1 = test_method_1_invoke_with_kwargs()
    test2 = test_method_2_func_call_with_kwargs()
    test3 = test_method_3_check_schema()
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Test 1 (tool.invoke + kwargs): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Test 2 (tool.func + kwargs):   {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Test 3 (schema check):         {'✅ PASS' if test3 else '❌ FAIL'}")
    print("="*70)
    
    if test2 and test3:
        print("\n✅ RECOMMENDATION: Use tool.func() directly + **kwargs")
        print("   - Context NOT in schema (LLM won't see it)")
        print("   - Context passed successfully to tool")
    elif test1 and test3:
        print("\n✅ RECOMMENDATION: Use tool.invoke() + **kwargs")
        print("   - Context NOT in schema (LLM won't see it)")
        print("   - Context passed successfully to tool")
    else:
        print("\n❌ No working method found")
    
    exit(0 if (test2 and test3) else 1)

