#!/usr/bin/env python3
"""
Multi-Agent System v0.3
=======================
Main entry point for the agent system with PM -> Planner-2 -> Code-Builder -> Critic -> Tool-Registry + Memory
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

def main():
    """Main entry point for the multi-agent system."""
    logger.info("Starting Multi-Agent System v0.3")
    
    # Initialize components
    memory = Memory()
    tool_registry = ToolRegistry()
    project_manager = ProjectManager()
    planner = Planner2()
    code_builder = CodeBuilder()
    critic = Critic()
    executor = Executor()
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Multi-Agent System > ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Shutting down Multi-Agent System")
                break
            
            # Process through the agent pipeline
            logger.info("Processing request through agent pipeline...")
            
            # 1. Project Manager: Add to backlog
            ticket = project_manager.add_to_backlog(user_input)
            logger.info(f"📋 PM: Added ticket {ticket['id']} to backlog")
            
            # 2. Planner-2: Decide on action
            plan = planner.create_plan(ticket, memory)
            logger.info(f"🧠 Planner-2: Created plan: {plan['action']}")
            
            # 3. Code Builder: Generate code if needed
            if plan['action'] == 'build_spec':
                spec = code_builder.build_spec(plan, memory)
                logger.info(f"🔨 Code-Builder: Generated spec for {spec['tool_name']}")
                
                # 4. Critic: Review the spec
                review = critic.review_spec(spec)
                logger.info(f"🔍 Critic: Review complete - {review['status']}")
                
                if review['status'] == 'approved':
                    # 5. Tool Registry: Register the tool
                    tool_registry.register_tool(spec)
                    logger.info(f"📚 Tool-Registry: Registered {spec['tool_name']}")
                    
                    # 6. Executor: Run tests and load
                    result = executor.execute_tool(spec)
                    logger.info(f"⚡ Executor: Tool execution - {result['status']}")
                    
                    # 7. Memory: Store the experience
                    memory.store_experience(ticket, plan, spec, review, result)
                    logger.info(f"🧠 Memory: Stored experience")
                    
                    if result['status'] == 'success':
                        print(f"\n✅ Successfully built and loaded: {spec['tool_name']}")
                        print(f"📝 Description: {spec['description']}")
                        if result.get('output'):
                            print(f"🔧 Output: {result['output']}")
                    else:
                        print(f"\n❌ Tool execution failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"\n⚠️ Spec rejected: {review.get('feedback', 'No feedback provided')}")
            else:
                print(f"\nℹ️ Action decided: {plan['action']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main() 