
"""
Base edge classes for LangGraph workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union

from citadel_langgraph.state.base import BaseState

# Define a generic type variable for state
S = TypeVar('S', bound=BaseState)


class BaseEdge(Generic[S], ABC):
    """
    Base class for all workflow edges.
    
    This class defines the interface for edges in a LangGraph workflow.
    """
    
    def __init__(self, name: str):
        """
        Initialize the edge.
        
        Args:
            name: Name of the edge.
        """
        self.name = name
    
    @abstractmethod
    def __call__(self, state: S) -> str:
        """
        Determine the next node based on the current state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        pass


class ConditionalEdge(BaseEdge[S]):
    """
    Edge that routes based on a condition.
    
    This class implements conditional routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        condition: Callable[[S], bool],
        true_node: str,
        false_node: str,
    ):
        """
        Initialize the conditional edge.
        
        Args:
            name: Name of the edge.
            condition: Function that returns True or False.
            true_node: Node to route to if condition is True.
            false_node: Node to route to if condition is False.
        """
        super().__init__(name)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node
    
    def __call__(self, state: S) -> str:
        """
        Determine the next node based on the condition.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        if self.condition(state):
            return self.true_node
        else:
            return self.false_node


class MultiRouteEdge(BaseEdge[S]):
    """
    Edge that routes to one of multiple nodes based on a routing function.
    
    This class implements multi-route routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        router: Callable[[S], str],
        routes: Dict[str, str],
        default_route: Optional[str] = None,
    ):
        """
        Initialize the multi-route edge.
        
        Args:
            name: Name of the edge.
            router: Function that returns a route key.
            routes: Dictionary mapping route keys to node names.
            default_route: Default node to route to if route key is not found.
        """
        super().__init__(name)
        self.router = router
        self.routes = routes
        self.default_route = default_route
    
    def __call__(self, state: S) -> str:
        """
        Determine the next node based on the router function.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        route_key = self.router(state)
        return self.routes.get(route_key, self.default_route)


class StatusBasedEdge(BaseEdge[S]):
    """
    Edge that routes based on the status field in the state.
    
    This class implements status-based routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        status_routes: Dict[str, str],
        default_route: Optional[str] = None,
    ):
        """
        Initialize the status-based edge.
        
        Args:
            name: Name of the edge.
            status_routes: Dictionary mapping status values to node names.
            default_route: Default node to route to if status is not found.
        """
        super().__init__(name)
        self.status_routes = status_routes
        self.default_route = default_route
    
    def __call__(self, state: S) -> str:
        """
        Determine the next node based on the status.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        status = state.get("status", "")
        return self.status_routes.get(status, self.default_route)


class ErrorHandlingEdge(BaseEdge[S]):
    """
    Edge that routes based on whether there is an error in the state.
    
    This class implements error handling routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        success_node: str,
        error_node: str,
    ):
        """
        Initialize the error handling edge.
        
        Args:
            name: Name of the edge.
            success_node: Node to route to if there is no error.
            error_node: Node to route to if there is an error.
        """
        super().__init__(name)
        self.success_node = success_node
        self.error_node = error_node
    
    def __call__(self, state: S) -> str:
        """
        Determine the next node based on whether there is an error.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        if state.get("error"):
            return self.error_node
        else:
            return self.success_node
