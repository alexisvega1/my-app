#!/usr/bin/env python3
"""
Tool Registry
============
Manages dynamic tool registration and loading.
"""

import os
import sys
import json
import importlib
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Manages dynamic tool registration and loading."""
    
    def __init__(self, registry_file: str = "tool_registry.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()
        self.loaded_tools = {}
        self._load_existing_tools()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the tool registry from file."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info(f"Creating new tool registry: {self.registry_file}")
            return {'tools': {}, 'metadata': {'created': datetime.now().isoformat()}}
        except json.JSONDecodeError:
            logger.warning(f"Corrupted registry file, starting fresh: {self.registry_file}")
            return {'tools': {}, 'metadata': {'created': datetime.now().isoformat()}}
    
    def _save_registry(self):
        """Save the tool registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_existing_tools(self):
        """Load existing tools from the registry."""
        for tool_name, tool_info in self.registry.get('tools', {}).items():
            if tool_info.get('status') == 'active':
                self._load_tool(tool_name, tool_info)
    
    def register_tool(self, spec: Dict[str, Any]) -> bool:
        """Register a new tool in the registry."""
        tool_name = spec['tool_name']
        
        # Check if tool already exists
        if tool_name in self.registry.get('tools', {}):
            logger.warning(f"Tool {tool_name} already registered")
            return False
        
        # Create tool entry
        tool_entry = {
            'name': tool_name,
            'file_path': spec['file_path'],
            'description': spec['description'],
            'tool_type': spec['tool_type'],
            'parameters': spec['parameters'],
            'imports': spec['imports'],
            'test_cases': spec['test_cases'],
            'status': 'active',
            'registered_at': datetime.now().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        
        # Add to registry
        if 'tools' not in self.registry:
            self.registry['tools'] = {}
        
        self.registry['tools'][tool_name] = tool_entry
        self._save_registry()
        
        # Load the tool
        success = self._load_tool(tool_name, tool_entry)
        
        logger.info(f"Registered tool: {tool_name}")
        return success
    
    def _load_tool(self, tool_name: str, tool_info: Dict[str, Any]) -> bool:
        """Load a tool into memory."""
        try:
            file_path = tool_info['file_path']
            
            # Ensure the file exists
            if not os.path.exists(file_path):
                logger.error(f"Tool file not found: {file_path}")
                return False
            
            # Add dyn_tools to Python path if not already there
            dyn_tools_path = os.path.dirname(file_path)
            if dyn_tools_path not in sys.path:
                sys.path.insert(0, dyn_tools_path)
            
            # Import the module
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            module = importlib.import_module(module_name)
            
            # Get the function
            if hasattr(module, tool_name):
                function = getattr(module, tool_name)
                self.loaded_tools[tool_name] = {
                    'function': function,
                    'info': tool_info
                }
                logger.info(f"Loaded tool: {tool_name}")
                return True
            else:
                logger.error(f"Function {tool_name} not found in module {module_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load tool {tool_name}: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a loaded tool function."""
        if tool_name in self.loaded_tools:
            # Update usage statistics
            self.registry['tools'][tool_name]['usage_count'] += 1
            self.registry['tools'][tool_name]['last_used'] = datetime.now().isoformat()
            self._save_registry()
            
            return self.loaded_tools[tool_name]['function']
        return None
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        tools = []
        for tool_name, tool_info in self.registry.get('tools', {}).items():
            tool_summary = {
                'name': tool_name,
                'description': tool_info['description'],
                'status': tool_info['status'],
                'usage_count': tool_info.get('usage_count', 0),
                'last_used': tool_info.get('last_used'),
                'loaded': tool_name in self.loaded_tools
            }
            tools.append(tool_summary)
        
        return tools
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name not in self.registry.get('tools', {}):
            logger.warning(f"Tool {tool_name} not found in registry")
            return False
        
        # Remove from loaded tools
        if tool_name in self.loaded_tools:
            del self.loaded_tools[tool_name]
        
        # Mark as inactive in registry
        self.registry['tools'][tool_name]['status'] = 'inactive'
        self.registry['tools'][tool_name]['unregistered_at'] = datetime.now().isoformat()
        self._save_registry()
        
        logger.info(f"Unregistered tool: {tool_name}")
        return True
    
    def reload_tool(self, tool_name: str) -> bool:
        """Reload a tool."""
        if tool_name not in self.registry.get('tools', {}):
            logger.warning(f"Tool {tool_name} not found in registry")
            return False
        
        # Remove from loaded tools
        if tool_name in self.loaded_tools:
            del self.loaded_tools[tool_name]
        
        # Reload
        tool_info = self.registry['tools'][tool_name]
        success = self._load_tool(tool_name, tool_info)
        
        if success:
            logger.info(f"Reloaded tool: {tool_name}")
        else:
            logger.error(f"Failed to reload tool: {tool_name}")
        
        return success
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        return self.registry.get('tools', {}).get(tool_name)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        tools = self.registry.get('tools', {})
        total = len(tools)
        active = len([t for t in tools.values() if t.get('status') == 'active'])
        loaded = len(self.loaded_tools)
        
        total_usage = sum(t.get('usage_count', 0) for t in tools.values())
        
        return {
            'total_tools': total,
            'active_tools': active,
            'loaded_tools': loaded,
            'total_usage': total_usage,
            'registry_file': self.registry_file
        } 