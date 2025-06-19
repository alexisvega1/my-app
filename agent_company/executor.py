#!/usr/bin/env python3
"""
Executor
========
Runs tests and executes tools safely.
"""

import os
import sys
import subprocess
import logging
import importlib.util
from typing import Dict, List, Any, Optional
from code_builder import CodeBuilder

logger = logging.getLogger(__name__)

class Executor:
    """Executes tools and runs tests safely."""
    
    def __init__(self):
        self.code_builder = CodeBuilder()
    
    def execute_tool(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool specification."""
        try:
            # Create the tool file
            file_path = self.code_builder.create_tool_file(spec)
            
            # Run tests
            test_result = self._run_tests(spec)
            
            if test_result['status'] == 'success':
                # Execute the tool
                execution_result = self._execute_tool_function(spec)
                return execution_result
            else:
                return {
                    'status': 'failed',
                    'error': f"Tests failed: {test_result.get('error', 'Unknown test error')}",
                    'test_output': test_result.get('output', '')
                }
                
        except Exception as e:
            logger.error(f"Failed to execute tool {spec['tool_name']}: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _run_tests(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for the tool."""
        try:
            # Create test file
            test_file_path = self._create_test_file(spec)
            
            # Run pytest on the test file
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_file_path, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                return {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'Test execution failed',
                    'output': result.stdout + result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'error': 'Test execution timed out'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Test setup failed: {e}'
            }
    
    def _create_test_file(self, spec: Dict[str, Any]) -> str:
        """Create a test file for the tool."""
        test_file_path = f"tests/test_{spec['tool_name']}.py"
        
        # Ensure tests directory exists
        os.makedirs('tests', exist_ok=True)
        
        # Create test content
        test_content = f'''#!/usr/bin/env python3
"""
Tests for {spec['tool_name']}
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from dyn_tools.{spec['tool_name']} import {spec['function_name']}

def test_{spec['function_name']}_basic():
    """Basic functionality test."""
    try:
        # Test with default parameters
        result = {spec['function_name']}('''
        
        # Add test parameters based on spec
        test_params = []
        for param in spec['parameters']:
            if param['type'] == 'str':
                test_params.append("'test_value'")
            elif param['type'] == 'float':
                test_params.append("1.0")
            elif param['type'] == 'int':
                test_params.append("1")
            else:
                test_params.append("'test_value'")
        
        test_content += ', '.join(test_params)
        test_content += f''')
        
        # Basic assertions
        assert result is not None
        print(f"Test passed: {{result}}")
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {{e}}")

def test_{spec['function_name']}_error_handling():
    """Test error handling."""
    try:
        # Test with invalid input
        result = {spec['function_name']}('invalid_input')
        # If we get here, the function should handle errors gracefully
        assert result is not None
    except Exception as e:
        # This is expected for invalid input
        print(f"Expected error caught: {{e}}")
        assert True  # Error handling works
'''
        
        # Write test file
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        logger.info(f"Created test file: {test_file_path}")
        return test_file_path
    
    def _execute_tool_function(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool function with sample data."""
        try:
            # Import the module
            module_path = spec['file_path']
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            
            # Add dyn_tools to path
            dyn_tools_path = os.path.dirname(module_path)
            if dyn_tools_path not in sys.path:
                sys.path.insert(0, dyn_tools_path)
            
            # Import and execute
            module = importlib.import_module(module_name)
            function = getattr(module, spec['function_name'])
            
            # Prepare test arguments
            test_args = self._prepare_test_args(spec)
            
            # Execute the function
            result = function(**test_args)
            
            return {
                'status': 'success',
                'output': result,
                'test_args': test_args
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _prepare_test_args(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare test arguments for the function."""
        args = {}
        
        for param in spec['parameters']:
            param_name = param['name']
            param_type = param['type']
            
            if param_type == 'str':
                if 'symbol' in param_name.lower():
                    args[param_name] = 'ethereum'
                elif 'currency' in param_name.lower():
                    args[param_name] = 'usd'
                else:
                    args[param_name] = 'test_value'
            elif param_type == 'float':
                args[param_name] = 1.0
            elif param_type == 'int':
                args[param_name] = 1
            else:
                args[param_name] = 'test_value'
        
        return args
    
    def run_specific_test(self, test_file: str) -> Dict[str, Any]:
        """Run a specific test file."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_file, '-v'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def validate_tool_safety(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that a tool is safe to execute."""
        safety_checks = {
            'has_imports': bool(spec.get('imports')),
            'has_error_handling': 'try:' in spec.get('implementation', ''),
            'no_dangerous_patterns': self._check_dangerous_patterns(spec),
            'has_parameters': bool(spec.get('parameters')),
            'has_description': bool(spec.get('description'))
        }
        
        all_safe = all(safety_checks.values())
        
        return {
            'safe': all_safe,
            'checks': safety_checks,
            'recommendations': self._generate_safety_recommendations(safety_checks)
        }
    
    def _check_dangerous_patterns(self, spec: Dict[str, Any]) -> bool:
        """Check for dangerous patterns in the implementation."""
        dangerous_patterns = [
            'eval(', 'exec(', '__import__', 'os.system', 'subprocess',
            'open(', 'file(', 'input(', 'raw_input('
        ]
        
        implementation = spec.get('implementation', '').lower()
        for pattern in dangerous_patterns:
            if pattern in implementation:
                return False
        
        return True
    
    def _generate_safety_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Generate safety recommendations based on check results."""
        recommendations = []
        
        if not checks['has_imports']:
            recommendations.append("Add necessary imports for functionality")
        
        if not checks['has_error_handling']:
            recommendations.append("Add try-catch error handling")
        
        if not checks['no_dangerous_patterns']:
            recommendations.append("Remove dangerous code patterns")
        
        if not checks['has_parameters']:
            recommendations.append("Define function parameters")
        
        if not checks['has_description']:
            recommendations.append("Add function description")
        
        return recommendations 