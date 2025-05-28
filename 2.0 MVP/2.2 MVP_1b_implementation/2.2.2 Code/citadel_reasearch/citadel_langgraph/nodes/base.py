
"""
Base node classes for LangGraph workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union

from citadel_langgraph.state.base import BaseState

# Define a generic type variable for state
S = TypeVar('S', bound=BaseState)


class BaseNode(Generic[S], ABC):
    """
    Base class for all workflow nodes.
    
    This class defines the interface for nodes in a LangGraph workflow.
    """
    
    def __init__(self, name: str):
        """
        Initialize the node.
        
        Args:
            name: Name of the node.
        """
        self.name = name
    
    @abstractmethod
    def __call__(self, state: S) -> S:
        """
        Execute the node's logic.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        pass


class FunctionNode(BaseNode[S]):
    """
    Node that executes a function.
    
    This class wraps a function as a node in a LangGraph workflow.
    """
    
    def __init__(self, name: str, func: Callable[[S], S]):
        """
        Initialize the function node.
        
        Args:
            name: Name of the node.
            func: Function to execute.
        """
        super().__init__(name)
        self.func = func
    
    def __call__(self, state: S) -> S:
        """
        Execute the function.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        return self.func(state)


class ConditionalNode(BaseNode[S]):
    """
    Node that executes different functions based on a condition.
    
    This class allows for conditional execution in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        condition: Callable[[S], bool],
        true_func: Callable[[S], S],
        false_func: Callable[[S], S],
    ):
        """
        Initialize the conditional node.
        
        Args:
            name: Name of the node.
            condition: Function that returns True or False.
            true_func: Function to execute if condition is True.
            false_func: Function to execute if condition is False.
        """
        super().__init__(name)
        self.condition = condition
        self.true_func = true_func
        self.false_func = false_func
    
    def __call__(self, state: S) -> S:
        """
        Execute the conditional logic.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        if self.condition(state):
            return self.true_func(state)
        else:
            return self.false_func(state)


class SequentialNode(BaseNode[S]):
    """
    Node that executes a sequence of functions.
    
    This class allows for sequential execution of multiple functions in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        funcs: List[Callable[[S], S]],
    ):
        """
        Initialize the sequential node.
        
        Args:
            name: Name of the node.
            funcs: List of functions to execute in sequence.
        """
        super().__init__(name)
        self.funcs = funcs
    
    def __call__(self, state: S) -> S:
        """
        Execute the sequence of functions.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        current_state = state
        for func in self.funcs:
            current_state = func(current_state)
        return current_state


class ErrorHandlingNode(BaseNode[S]):
    """
    Node that handles errors in function execution.
    
    This class wraps a function with error handling in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        func: Callable[[S], S],
        error_handler: Callable[[S, Exception], S],
    ):
        """
        Initialize the error handling node.
        
        Args:
            name: Name of the node.
            func: Function to execute.
            error_handler: Function to handle errors.
        """
        super().__init__(name)
        self.func = func
        self.error_handler = error_handler
    
    def __call__(self, state: S) -> S:
        """
        Execute the function with error handling.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        try:
            return self.func(state)
        except Exception as e:
            return self.error_handler(state, e)
