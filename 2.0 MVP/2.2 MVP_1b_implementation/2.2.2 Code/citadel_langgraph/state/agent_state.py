
"""
Specialized state classes for agent workflows.
"""

from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union
from datetime import datetime

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)

from .base import BaseState, AgentState, create_agent_state


class ReActAgentState(AgentState):
    """
    State class for ReAct agent workflows.
    
    This class extends AgentState to include ReAct-specific fields.
    """
    
    # Current thought process
    thought: Optional[str]
    
    # Current action
    action: Optional[str]
    
    # Current action input
    action_input: Optional[Dict[str, Any]]
    
    # Observation from the last action
    observation: Optional[str]


def create_react_agent_state(
    system_message: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ReActAgentState:
    """
    Create a new ReAct agent state with default values.
    
    Args:
        system_message: Optional system message to include in the state.
        tools: Optional list of tools available to the agent.
        metadata: Optional metadata to include in the state.
        
    Returns:
        A new ReActAgentState instance.
    """
    agent_state = create_agent_state(system_message, tools, metadata)
    return ReActAgentState(
        **agent_state,
        thought=None,
        action=None,
        action_input=None,
        observation=None,
    )


class MultiAgentState(BaseState):
    """
    State class for multi-agent workflows.
    
    This class extends BaseState to include multi-agent-specific fields.
    """
    
    # Agent states
    agent_states: Dict[str, AgentState]
    
    # Shared memory
    shared_memory: Dict[str, Any]
    
    # Current active agent
    active_agent: Optional[str]
    
    # Agent execution history
    execution_history: List[Dict[str, Any]]


def create_multi_agent_state(
    agent_configs: Dict[str, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> MultiAgentState:
    """
    Create a new multi-agent state with default values.
    
    Args:
        agent_configs: Dictionary mapping agent IDs to their configurations.
        metadata: Optional metadata to include in the state.
        
    Returns:
        A new MultiAgentState instance.
    """
    base_state = BaseState(
        workflow_id=str(datetime.utcnow().timestamp()),
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
        status="initialized",
        error=None,
        metadata=metadata or {},
    )
    
    # Create agent states
    agent_states = {}
    for agent_id, config in agent_configs.items():
        agent_states[agent_id] = create_agent_state(
            system_message=config.get("system_message"),
            tools=config.get("tools"),
            metadata=config.get("metadata"),
        )
    
    return MultiAgentState(
        **base_state,
        agent_states=agent_states,
        shared_memory={},
        active_agent=None,
        execution_history=[],
    )
