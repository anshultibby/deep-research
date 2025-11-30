"""Test to inspect what schema LangChain generates for the LLM."""
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from models import AgentContext
import json


@tool
def test_tool_with_default(
    query: Annotated[str, "A test query"],
    context: AgentContext = None
) -> str:
    """A test tool that accepts context as a default parameter."""
    return f"Query: {query}, Has context: {context is not None}"


@tool
def test_tool_with_injected(
    query: Annotated[str, "A test query"],
    context: Annotated[AgentContext, InjectedToolArg]
) -> str:
    """A test tool that uses InjectedToolArg."""
    return f"Query: {query}, Has context: {context is not None}"


@tool
def test_tool_without_context(
    query: Annotated[str, "A test query"]
) -> str:
    """A test tool without context."""
    return f"Query: {query}"


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SCHEMA INSPECTION - Testing InjectedToolArg")
    print("="*70)
    
    print("\n1. Tool WITH context (default value approach):")
    print("-" * 70)
    schema1 = test_tool_with_default.get_input_schema().model_json_schema()
    properties1 = list(schema1.get('properties', {}).keys())
    required1 = schema1.get('required', [])
    print(f"Properties: {properties1}")
    print(f"Required: {required1}")
    
    print("\n2. Tool WITH context (InjectedToolArg approach):")
    print("-" * 70)
    schema2 = test_tool_with_injected.get_input_schema().model_json_schema()
    properties2 = list(schema2.get('properties', {}).keys())
    required2 = schema2.get('required', [])
    print(f"Properties: {properties2}")
    print(f"Required: {required2}")
    
    print("\n3. Tool WITHOUT context:")
    print("-" * 70)
    schema3 = test_tool_without_context.get_input_schema().model_json_schema()
    properties3 = list(schema3.get('properties', {}).keys())
    required3 = schema3.get('required', [])
    print(f"Properties: {properties3}")
    print(f"Required: {required3}")
    
    print("\n" + "="*70)
    print("ANALYSIS:")
    print("="*70)
    
    # Check if context is in schema
    context_in_default = 'context' in properties1
    context_in_injected = 'context' in properties2
    context_in_none = 'context' in properties3
    
    print(f"\n1. Default value approach:")
    if context_in_default:
        print(f"   ❌ FAIL: context is in schema (properties: {properties1})")
    else:
        print(f"   ✅ PASS: context NOT in schema")
    
    print(f"\n2. InjectedToolArg approach:")
    if context_in_injected:
        print(f"   ❌ FAIL: context is in schema (properties: {properties2})")
    else:
        print(f"   ✅ PASS: context NOT in schema")
    
    print(f"\n3. No context:")
    if context_in_none:
        print(f"   ❌ FAIL: context somehow in schema (properties: {properties3})")
    else:
        print(f"   ✅ PASS: context NOT in schema")
    
    print("\n" + "="*70)
    
    if not context_in_injected:
        print("✅ InjectedToolArg successfully excludes context from LLM schema!")
    else:
        print("❌ InjectedToolArg did NOT work - context still in schema!")
    
    print("="*70)

