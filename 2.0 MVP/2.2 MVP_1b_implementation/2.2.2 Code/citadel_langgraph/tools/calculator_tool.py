
"""
Calculator tool for agent workflows.
"""

import logging
import math
from typing import Dict, Any, Optional, Union

from citadel_core.logging import get_logger
from .tool_registry import BaseTool


class CalculatorTool(BaseTool):
    """
    Tool for performing mathematical calculations.
    
    This tool allows agents to perform various mathematical operations.
    """
    
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the calculator tool.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger or get_logger("citadel.langgraph.tools.calculator")
    
    def __call__(self, expression: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: The mathematical expression to evaluate.
            **kwargs: Additional parameters.
            
        Returns:
            The result of the calculation.
        """
        self.logger.info(f"Evaluating expression: {expression}")
        
        try:
            # Create a safe evaluation environment with limited functions
            safe_env = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "math": math,
            }
            
            # Evaluate the expression in the safe environment
            result = eval(expression, {"__builtins__": {}}, safe_env)
            
            self.logger.info(f"Expression evaluated: {expression} = {result}")
            
            return {
                "expression": expression,
                "result": result,
            }
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {str(e)}")
            return {
                "expression": expression,
                "error": str(e),
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary representation.
        
        Returns:
            Dictionary representation of the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                },
            },
        }
