#!/usr/bin/env python3
"""
Critic
======
Reviews and validates tool specifications for quality and safety.
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class Critic:
    """Reviews and validates tool specifications."""
    
    def __init__(self):
        self.safety_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'os\.system',
            r'subprocess',
            r'import\s+os',
            r'import\s+subprocess'
        ]
        
        self.quality_checks = {
            'has_docstring': True,
            'has_error_handling': True,
            'has_type_hints': False,  # Optional for now
            'has_meaningful_name': True,
            'has_parameters': True
        }
    
    def review_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Review a tool specification and provide feedback."""
        issues = []
        warnings = []
        
        # Check for safety issues
        safety_issues = self._check_safety(spec)
        issues.extend(safety_issues)
        
        # Check for quality issues
        quality_issues = self._check_quality(spec)
        issues.extend(quality_issues)
        
        # Check for potential improvements
        improvements = self._suggest_improvements(spec)
        warnings.extend(improvements)
        
        # Determine overall status
        status = 'approved' if not issues else 'rejected'
        
        review = {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'score': self._calculate_score(spec, issues, warnings),
            'feedback': self._generate_feedback(issues, warnings)
        }
        
        logger.info(f"Review complete for {spec['tool_name']}: {status}")
        return review
    
    def _check_safety(self, spec: Dict[str, Any]) -> List[str]:
        """Check for safety issues in the specification."""
        issues = []
        implementation = spec.get('implementation', '')
        
        for pattern in self.safety_patterns:
            if re.search(pattern, implementation, re.IGNORECASE):
                issues.append(f"Safety issue: Found potentially dangerous pattern '{pattern}'")
        
        # Check for network requests without proper error handling
        if 'requests.get(' in implementation and 'raise_for_status()' not in implementation:
            issues.append("Safety issue: Network request without proper error handling")
        
        # Check for hardcoded URLs or credentials
        if re.search(r'https?://[^\s]+', implementation):
            if not any(word in implementation.lower() for word in ['api.coingecko', 'example.com', 'test.com']):
                issues.append("Warning: Hardcoded URL found - consider making it configurable")
        
        return issues
    
    def _check_quality(self, spec: Dict[str, Any]) -> List[str]:
        """Check for quality issues in the specification."""
        issues = []
        
        # Check if function name is meaningful
        if len(spec['function_name']) < 3:
            issues.append("Quality issue: Function name too short")
        
        if not re.match(r'^[a-z_][a-z0-9_]*$', spec['function_name']):
            issues.append("Quality issue: Function name contains invalid characters")
        
        # Check if parameters are well-defined
        if not spec.get('parameters'):
            issues.append("Quality issue: No parameters defined")
        
        # Check if description is adequate
        if len(spec.get('description', '')) < 10:
            issues.append("Quality issue: Description too short")
        
        # Check if implementation has basic structure
        implementation = spec.get('implementation', '')
        if not implementation.strip():
            issues.append("Quality issue: Empty implementation")
        
        if 'try:' not in implementation and 'except' not in implementation:
            issues.append("Quality issue: No error handling in implementation")
        
        return issues
    
    def _suggest_improvements(self, spec: Dict[str, Any]) -> List[str]:
        """Suggest improvements for the specification."""
        suggestions = []
        
        # Check for type hints
        if not any('type' in param for param in spec.get('parameters', [])):
            suggestions.append("Consider adding type hints to parameters")
        
        # Check for comprehensive docstring
        if len(spec.get('description', '')) < 50:
            suggestions.append("Consider providing a more detailed description")
        
        # Check for test coverage
        if not spec.get('test_cases'):
            suggestions.append("Consider adding test cases")
        
        # Check for logging
        implementation = spec.get('implementation', '')
        if 'logging' not in implementation and 'print' not in implementation:
            suggestions.append("Consider adding logging for debugging")
        
        return suggestions
    
    def _calculate_score(self, spec: Dict[str, Any], issues: List[str], warnings: List[str]) -> float:
        """Calculate a quality score for the specification."""
        base_score = 100.0
        
        # Deduct points for issues
        base_score -= len(issues) * 20
        
        # Deduct points for warnings
        base_score -= len(warnings) * 5
        
        # Bonus points for good practices
        if spec.get('test_cases'):
            base_score += 10
        
        if len(spec.get('description', '')) > 50:
            base_score += 5
        
        if spec.get('parameters'):
            base_score += 5
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_feedback(self, issues: List[str], warnings: List[str]) -> str:
        """Generate human-readable feedback."""
        if not issues and not warnings:
            return "Specification looks good! Ready for implementation."
        
        feedback_parts = []
        
        if issues:
            feedback_parts.append("Issues to fix:")
            for issue in issues:
                feedback_parts.append(f"  - {issue}")
        
        if warnings:
            feedback_parts.append("Suggestions for improvement:")
            for warning in warnings:
                feedback_parts.append(f"  - {warning}")
        
        return "\n".join(feedback_parts)
    
    def is_safe_for_execution(self, spec: Dict[str, Any]) -> bool:
        """Check if the specification is safe for execution."""
        safety_issues = self._check_safety(spec)
        return len(safety_issues) == 0
    
    def get_review_summary(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the review."""
        review = self.review_spec(spec)
        
        return {
            'tool_name': spec['tool_name'],
            'status': review['status'],
            'score': review['score'],
            'issue_count': len(review['issues']),
            'warning_count': len(review['warnings']),
            'safe_for_execution': self.is_safe_for_execution(spec)
        } 