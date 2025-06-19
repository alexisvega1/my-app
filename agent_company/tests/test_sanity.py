#!/usr/bin/env python3
"""
Sanity tests for the multi-agent system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from project_manager import load_backlog

def test_backlog_loads():
    """Test that the backlog can be loaded."""
    backlog = load_backlog()
    assert isinstance(backlog, list)
    print("✅ Backlog loads successfully")

def test_project_manager_creation():
    """Test that ProjectManager can be created."""
    from project_manager import ProjectManager
    pm = ProjectManager()
    assert pm is not None
    print("✅ ProjectManager creates successfully")

def test_memory_creation():
    """Test that Memory can be created."""
    from memory import Memory
    memory = Memory()
    assert memory is not None
    print("✅ Memory creates successfully")

def test_tool_registry_creation():
    """Test that ToolRegistry can be created."""
    from tool_registry import ToolRegistry
    tr = ToolRegistry()
    assert tr is not None
    print("✅ ToolRegistry creates successfully")

if __name__ == "__main__":
    test_backlog_loads()
    test_project_manager_creation()
    test_memory_creation()
    test_tool_registry_creation()
    print("\n🎉 All sanity tests passed!") 