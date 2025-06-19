#!/usr/bin/env python3
"""
Code Builder
===========
Generates tool specifications and code based on plans and memory.
"""

import os
import re
import logging
from typing import Dict, List, Any
from memory import Memory

logger = logging.getLogger(__name__)

class CodeBuilder:
    """Generates tool specifications and code."""
    
    def __init__(self):
        self.tool_templates = {
            'data_fetch': {
                'imports': ['requests', 'json'],
                'template': '''
def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {param_docs}
    
    Returns:
        {return_type}: {return_description}
    """
    try:
        {implementation}
        return result
    except Exception as e:
        raise Exception(f"Error in {function_name}: {{e}}")
'''
            },
            'calculation': {
                'imports': ['math'],
                'template': '''
def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {param_docs}
    
    Returns:
        {return_type}: {return_description}
    """
    try:
        {implementation}
        return result
    except Exception as e:
        raise Exception(f"Error in {function_name}: {{e}}")
'''
            },
            'utility': {
                'imports': [],
                'template': '''
def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {param_docs}
    
    Returns:
        {return_type}: {return_description}
    """
    try:
        {implementation}
        return result
    except Exception as e:
        raise Exception(f"Error in {function_name}: {{e}}")
'''
            }
        }
    
    def build_spec(self, plan: Dict[str, Any], memory: Memory) -> Dict[str, Any]:
        """Build a tool specification based on the plan."""
        description = plan['description']
        
        # Analyze the description to determine tool type and requirements
        tool_type = self._determine_tool_type(description)
        function_name = self._generate_function_name(description)
        parameters = self._extract_parameters(description)
        implementation = self._generate_implementation(description, tool_type)
        
        # Create the specification
        spec = {
            'tool_name': function_name,
            'tool_type': tool_type,
            'description': description,
            'function_name': function_name,
            'parameters': parameters,
            'implementation': implementation,
            'imports': self.tool_templates[tool_type]['imports'],
            'test_cases': self._generate_test_cases(function_name, parameters),
            'file_path': f"dyn_tools/{function_name}.py"
        }
        
        logger.info(f"Built spec for {function_name}")
        return spec
    
    def _determine_tool_type(self, description: str) -> str:
        """Determine the type of tool based on description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['fetch', 'get', 'download', 'api', 'url', 'price']):
            return 'data_fetch'
        elif any(word in description_lower for word in ['calculate', 'compute', 'math', 'formula']):
            return 'calculation'
        else:
            return 'utility'
    
    def _generate_function_name(self, description: str) -> str:
        """Generate a function name from the description."""
        # Extract key words
        words = re.findall(r'\\b\\w+\\b', description.lower())
        
        # Filter out common words
        common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'that', 'this', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall'}
        key_words = [word for word in words if word not in common_words and len(word) > 2]
        
        if not key_words:
            return 'process_data'
        
        # Create function name
        function_name = '_'.join(key_words[:3])  # Use up to 3 key words
        return function_name
    
    def _extract_parameters(self, description: str) -> List[Dict[str, str]]:
        """Extract parameters from the description."""
        parameters = []
        
        # Look for common parameter patterns
        if 'price' in description.lower():
            parameters.append({
                'name': 'symbol',
                'type': 'str',
                'description': 'The trading symbol (e.g., "ETH-USD")'
            })
        
        if 'convert' in description.lower():
            parameters.append({
                'name': 'amount',
                'type': 'float',
                'description': 'The amount to convert'
            })
            parameters.append({
                'name': 'from_currency',
                'type': 'str',
                'description': 'Source currency'
            })
            parameters.append({
                'name': 'to_currency',
                'type': 'str',
                'description': 'Target currency'
            })
        
        # Default parameter if none found
        if not parameters:
            parameters.append({
                'name': 'data',
                'type': 'str',
                'description': 'Input data to process'
            })
        
        return parameters
    
    def _generate_implementation(self, description: str, tool_type: str) -> str:
        """Generate implementation code based on description and tool type."""
        description_lower = description.lower()
        
        if tool_type == 'data_fetch':
            if 'price' in description_lower:
                return '''
        import requests
        import json
        
        # Validate input
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Make API request
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={{symbol.lower()}}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        if symbol.lower() not in data:
            raise ValueError(f"Symbol {{symbol}} not found in API response")
        
        result = data[symbol.lower()]['usd']
        return float(result)
'''
            else:
                return '''
        import requests
        import json
        
        # Validate input
        if not data or not isinstance(data, str):
            raise ValueError("Data must be a non-empty string")
        
        # Make request
        response = requests.get(data, timeout=10)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result
'''
        
        elif tool_type == 'calculation':
            if 'convert' in description_lower:
                return '''
        # Validate inputs
        if not isinstance(amount, (int, float)):
            raise ValueError("Amount must be a number")
        if not isinstance(from_currency, str) or not isinstance(to_currency, str):
            raise ValueError("Currencies must be strings")
        
        # Simple conversion (in real implementation, use actual exchange rates)
        conversion_rates = {{
            'usd_to_eur': 0.85,
            'eur_to_usd': 1.18,
            'usd_to_gbp': 0.73,
            'gbp_to_usd': 1.37
        }}
        
        key = f"{{from_currency.lower()}}_to_{{to_currency.lower()}}"
        if key in conversion_rates:
            result = amount * conversion_rates[key]
        else:
            result = amount  # Default to no conversion
        
        return float(result)
'''
            else:
                return '''
        # Validate input
        if not data or not isinstance(data, str):
            raise ValueError("Data must be a non-empty string")
        
        # Safe evaluation (in production, use ast.literal_eval or similar)
        try:
            result = eval(data)
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {{e}}")
'''
        
        else:  # utility
            return '''
        # Validate input
        if not data or not isinstance(data, str):
            raise ValueError("Data must be a non-empty string")
        
        # Generic processing
        result = data.upper()
        return result
'''
    
    def _generate_test_cases(self, function_name: str, parameters: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Generate test cases for the function."""
        test_cases = []
        
        # Generate basic test case
        test_input = {}
        for param in parameters:
            if param['type'] == 'str':
                test_input[param['name']] = 'test_value'
            elif param['type'] == 'float':
                test_input[param['name']] = 1.0
            elif param['type'] == 'int':
                test_input[param['name']] = 1
            else:
                test_input[param['name']] = 'test_value'
        
        test_cases.append({
            'name': f'test_{function_name}_basic',
            'input': test_input,
            'expected_type': 'any',
            'description': f'Basic test for {function_name}'
        })
        
        return test_cases
    
    def create_tool_file(self, spec: Dict[str, Any]) -> str:
        """Create the actual tool file from specification."""
        # Ensure dyn_tools directory exists
        os.makedirs('dyn_tools', exist_ok=True)
        
        # Generate the code
        template = self.tool_templates[spec['tool_type']]['template']
        
        # Prepare template variables
        param_docs = []
        param_list = []
        for param in spec['parameters']:
            param_list.append(param['name'])
            param_docs.append(f"{param['name']} ({param['type']}): {param['description']}")
        
        template_vars = {
            'function_name': spec['function_name'],
            'parameters': ', '.join(param_list),
            'description': spec['description'],
            'param_docs': '\n        '.join(param_docs),
            'return_type': 'any',
            'return_description': 'The processed result',
            'implementation': spec['implementation']
        }
        
        # Generate the code
        code = template.format(**template_vars)
        
        # Add imports
        if spec['imports']:
            import_lines = '\n'.join([f'import {imp}' for imp in spec['imports']])
            code = f"{import_lines}\n\n{code}"
        
        # Write to file
        with open(spec['file_path'], 'w') as f:
            f.write(code)
        
        logger.info(f"Created tool file: {spec['file_path']}")
        return spec['file_path'] 