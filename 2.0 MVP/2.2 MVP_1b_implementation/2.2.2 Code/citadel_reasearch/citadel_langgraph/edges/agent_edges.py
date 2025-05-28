
"""
Agent edges for LangGraph workflows.
"""

from typing import Any, Dict, List, Optional, Callable

from citadel_langgraph.state.agent_state import (
    AgentState,
    ReActAgentState,
    MultiAgentState,
)
from .base import BaseEdge, ConditionalEdge, MultiRouteEdge


class ReActAgentStepEdge(BaseEdge[ReActAgentState]):
    """
    Edge that routes based on the current step in a ReAct agent workflow.
    
    This class implements step-based routing in a ReAct agent workflow.
    """
    
    def __init__(
        self,
        name: str,
        step_routes: Dict[str, str],
        default_route: Optional[str] = None,
    ):
        """
        Initialize the ReAct agent step edge.
        
        Args:
            name: Name of the edge.
            step_routes: Dictionary mapping step names to node names.
            default_route: Default node to route to if step is not found.
        """
        super().__init__(name)
        self.step_routes = step_routes
        self.default_route = default_route
    
    def __call__(self, state: ReActAgentState) -> str:
        """
        Determine the next node based on the current step.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        current_step = state.get("current_step", "")
        return self.step_routes.get(current_step, self.default_route)


class ActionTypeEdge(BaseEdge[ReActAgentState]):
    """
    Edge that routes based on the action type in a ReAct agent workflow.
    
    This class implements action-based routing in a ReAct agent workflow.
    """
    
    def __init__(
        self,
        name: str,
        action_routes: Dict[str, str],
        default_route: Optional[str] = None,
    ):
        """
        Initialize the action type edge.
        
        Args:
            name: Name of the edge.
            action_routes: Dictionary mapping action names to node names.
            default_route: Default node to route to if action is not found.
        """
        super().__init__(name)
        self.action_routes = action_routes
        self.default_route = default_route
    
    def __call__(self, state: ReActAgentState) -> str:
        """
        Determine the next node based on the action type.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        action = state.get("action", "")
        return self.action_routes.get(action, self.default_route)


class FinalAnswerEdge(ConditionalEdge[ReActAgentState]):
    """
    Edge that routes based on whether the agent has reached a final answer.
    
    This class implements final answer detection in a ReAct agent workflow.
    """
    
    def __init__(
        self,
        name: str,
        final_node: str,
        continue_node: str,
    ):
        """
        Initialize the final answer edge.
        
        Args:
            name: Name of the edge.
            final_node: Node to route to if a final answer is detected.
            continue_node: Node to route to if no final answer is detected.
        """
        super().__init__(
            name=name,
            condition=self._is_final_answer,
            true_node=final_node,
            false_node=continue_node,
        )
    
    def _is_final_answer(self, state: ReActAgentState) -> bool:
        """
        Check if the agent has reached a final answer.
        
        Args:
            state: Current workflow state.
            
        Returns:
            True if a final answer is detected, False otherwise.
        """
        thought = state.get("thought", "")
        return (
            "final answer" in thought.lower() or
            "i've solved the task" in thought.lower() or
            "task complete" in thought.lower()
        )


class MultiAgentCoordinationEdge(BaseEdge[MultiAgentState]):
    """
    Edge that routes based on the active agent in a multi-agent workflow.
    
    This class implements agent-based routing in a multi-agent workflow.
    """
    
    def __init__(
        self,
        name: str,
        agent_routes: Dict[str, str],
        default_route: Optional[str] = None,
    ):
        """
        Initialize the multi-agent coordination edge.
        
        Args:
            name: Name of the edge.
            agent_routes: Dictionary mapping agent IDs to node names.
            default_route: Default node to route to if agent ID is not found.
        """
        super().__init__(name)
        self.agent_routes = agent_routes
        self.default_route = default_route
    
    def __call__(self, state: MultiAgentState) -> str:
        """
        Determine the next node based on the active agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        active_agent = state.get("active_agent", "")
        return self.agent_routes.get(active_agent, self.default_route)


class TaskCompletionEdge(ConditionalEdge[MultiAgentState]):
    """
    Edge that routes based on whether the multi-agent task is complete.
    
    This class implements task completion detection in a multi-agent workflow.
    """
    
    def __init__(
        self,
        name: str,
        completion_condition: Callable[[MultiAgentState], bool],
        complete_node: str,
        incomplete_node: str,
    ):
        """
        Initialize the task completion edge.
        
        Args:
            name: Name of the edge.
            completion_condition: Function that determines if the task is complete.
            complete_node: Node to route to if the task is complete.
            incomplete_node: Node to route to if the task is incomplete.
        """
        super().__init__(
            name=name,
            condition=completion_condition,
            true_node=complete_node,
            false_node=incomplete_node,
        )
