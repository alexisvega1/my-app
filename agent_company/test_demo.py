#!/usr/bin/env python3
"""
Demo script to test the multi-agent system.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from project_manager import ProjectManager
from planner2 import Planner2
from code_builder import CodeBuilder
from critic import Critic
from tool_registry import ToolRegistry
from memory import Memory
from executor import Executor

def demo_eth_price_tool():
    """Demo the ETH price tool creation."""
    print("🚀 Starting Multi-Agent System Demo")
    print("=" * 50)
    
    # Initialize components
    memory = Memory()
    tool_registry = ToolRegistry()
    project_manager = ProjectManager()
    planner = Planner2()
    code_builder = CodeBuilder()
    critic = Critic()
    executor = Executor()
    
    # Test input
    user_input = "Build a function that fetches current ETH-USD price and returns it as float"
    print(f"📝 User Input: {user_input}")
    print()
    
    # 1. Project Manager: Add to backlog
    ticket = project_manager.add_to_backlog(user_input)
    print(f"📋 PM: Added ticket {ticket['id']} to backlog")
    
    # 2. Planner-2: Decide on action
    plan = planner.create_plan(ticket, memory)
    print(f"🧠 Planner-2: Created plan: {plan['action']}")
    print(f"   Reasoning: {plan['reasoning']}")
    print(f"   Confidence: {plan['confidence']:.2f}")
    
    # 3. Code Builder: Generate code if needed
    if plan['action'] == 'build_spec':
        spec = code_builder.build_spec(plan, memory)
        print(f"🔨 Code-Builder: Generated spec for {spec['tool_name']}")
        print(f"   Tool Type: {spec['tool_type']}")
        print(f"   Parameters: {len(spec['parameters'])}")
        
        # 4. Critic: Review the spec
        review = critic.review_spec(spec)
        print(f"🔍 Critic: Review complete - {review['status']}")
        print(f"   Score: {review['score']:.1f}/100")
        if review['issues']:
            print(f"   Issues: {len(review['issues'])}")
        if review['warnings']:
            print(f"   Warnings: {len(review['warnings'])}")
        
        if review['status'] == 'approved':
            # 5. Tool Registry: Register the tool
            success = tool_registry.register_tool(spec)
            print(f"📚 Tool-Registry: Registered {spec['tool_name']} - {'Success' if success else 'Failed'}")
            
            # 6. Executor: Run tests and load
            result = executor.execute_tool(spec)
            print(f"⚡ Executor: Tool execution - {result['status']}")
            
            # 7. Memory: Store the experience
            memory.store_experience(ticket, plan, spec, review, result)
            print(f"🧠 Memory: Stored experience")
            
            if result['status'] == 'success':
                print(f"\n✅ Successfully built and loaded: {spec['tool_name']}")
                print(f"📝 Description: {spec['description']}")
                if result.get('output'):
                    print(f"🔧 Output: {result['output']}")
                
                # Test the tool
                tool_function = tool_registry.get_tool(spec['tool_name'])
                if tool_function:
                    try:
                        test_result = tool_function('ethereum')
                        print(f"🧪 Test Result: {test_result}")
                    except Exception as e:
                        print(f"❌ Test failed: {e}")
            else:
                print(f"\n❌ Tool execution failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n⚠️ Spec rejected: {review.get('feedback', 'No feedback provided')}")
    else:
        print(f"\nℹ️ Action decided: {plan['action']}")
    
    print("\n" + "=" * 50)
    print("📊 System Statistics:")
    
    # Show backlog stats
    backlog_stats = project_manager.get_backlog_stats()
    print(f"   Backlog: {backlog_stats['total']} total, {backlog_stats['pending']} pending")
    
    # Show memory stats
    memory_stats = memory.get_memory_stats()
    print(f"   Memory: {memory_stats['total_experiences']} experiences, {memory_stats['success_rate']:.1%} success rate")
    
    # Show registry stats
    registry_stats = tool_registry.get_registry_stats()
    print(f"   Tools: {registry_stats['total_tools']} total, {registry_stats['loaded_tools']} loaded")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    demo_eth_price_tool() 