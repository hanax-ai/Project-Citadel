
"""
UI-aware state classes for LangGraph workflows.

This module provides state classes that are aware of the UI and can emit events
for state changes, enabling real-time monitoring and visualization of workflow execution.
"""

import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union, Set, cast
from datetime import datetime
import uuid
import copy

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    BaseMessage,
)

from citadel_langgraph.state.base import (
    BaseState, 
    MessageState, 
    DocumentState, 
    AgentState,
    create_base_state,
    create_message_state,
    create_agent_state
)
from citadel_langgraph.events.emitter import (
    emit_state_snapshot,
    emit_event,
    emit_message_start,
    emit_message_content,
    emit_message_end
)
from citadel_frontend.protocol.ag_ui import EventType


class UIState(BaseState):
    """
    UI-aware base state class for LangGraph workflows.
    
    This class extends BaseState to include UI-specific fields and methods
    for emitting events to the UI.
    """
    
    # UI context information
    ui_context: Dict[str, Any]
    
    # Flag to indicate if events should be emitted
    emit_events: bool
    
    # Message IDs for tracking message events
    message_ids: Dict[str, str]


def create_ui_state(
    metadata: Optional[Dict[str, Any]] = None,
    ui_context: Optional[Dict[str, Any]] = None,
    emit_events: bool = True
) -> UIState:
    """
    Create a new UI-aware base state with default values.
    
    Args:
        metadata: Optional metadata to include in the state.
        ui_context: Optional UI context information.
        emit_events: Whether to emit events for state changes.
        
    Returns:
        A new UIState instance.
    """
    base_state = create_base_state(metadata)
    return UIState(
        **base_state,
        ui_context=ui_context or {},
        emit_events=emit_events,
        message_ids={}
    )


class UIMessageState(MessageState, UIState):
    """
    UI-aware message state class for LangGraph workflows.
    
    This class extends MessageState to include UI-specific fields and methods
    for emitting events to the UI.
    """
    pass


def create_ui_message_state(
    system_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    ui_context: Optional[Dict[str, Any]] = None,
    emit_events: bool = True
) -> UIMessageState:
    """
    Create a new UI-aware message state with default values.
    
    Args:
        system_message: Optional system message to include in the state.
        metadata: Optional metadata to include in the state.
        ui_context: Optional UI context information.
        emit_events: Whether to emit events for state changes.
        
    Returns:
        A new UIMessageState instance.
    """
    message_state = create_message_state(system_message, metadata)
    return UIMessageState(
        **message_state,
        ui_context=ui_context or {},
        emit_events=emit_events,
        message_ids={}
    )


class UIAgentState(AgentState, UIState):
    """
    UI-aware agent state class for LangGraph workflows.
    
    This class extends AgentState to include UI-specific fields and methods
    for emitting events to the UI.
    """
    pass


def create_ui_agent_state(
    system_message: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    ui_context: Optional[Dict[str, Any]] = None,
    emit_events: bool = True
) -> UIAgentState:
    """
    Create a new UI-aware agent state with default values.
    
    Args:
        system_message: Optional system message to include in the state.
        tools: Optional list of tools available to the agent.
        metadata: Optional metadata to include in the state.
        ui_context: Optional UI context information.
        emit_events: Whether to emit events for state changes.
        
    Returns:
        A new UIAgentState instance.
    """
    agent_state = create_agent_state(system_message, tools, metadata)
    return UIAgentState(
        **agent_state,
        ui_context=ui_context or {},
        emit_events=emit_events,
        message_ids={}
    )


def emit_state_update(state: UIState) -> None:
    """
    Emit a state update event for the given state.
    
    Args:
        state: The state to emit an update for.
    """
    if not state.get("emit_events", True):
        return
    
    # Create a serializable copy of the state
    serializable_state = prepare_state_for_serialization(state)
    
    # Emit the state snapshot event
    emit_state_snapshot(serializable_state)


def prepare_state_for_serialization(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a state for serialization to JSON.
    
    Args:
        state: The state to prepare.
        
    Returns:
        A serializable copy of the state.
    """
    # Create a deep copy of the state to avoid modifying the original
    serializable_state = copy.deepcopy(state)
    
    # Handle message serialization
    if "messages" in serializable_state:
        serializable_state["messages"] = [
            {
                "type": msg.__class__.__name__,
                "content": msg.content,
                "additional_kwargs": msg.additional_kwargs
            }
            for msg in serializable_state["messages"]
        ]
    
    # Remove any non-serializable objects
    for key in list(serializable_state.keys()):
        try:
            # Test if the value is JSON serializable
            json.dumps(serializable_state[key])
        except (TypeError, OverflowError):
            # If not, convert to string or remove
            try:
                serializable_state[key] = str(serializable_state[key])
            except:
                del serializable_state[key]
    
    return serializable_state


def add_message_with_events(
    state: UIMessageState,
    message: BaseMessage,
    emit_events: Optional[bool] = None
) -> UIMessageState:
    """
    Add a message to the state and emit appropriate events.
    
    Args:
        state: The state to add the message to.
        message: The message to add.
        emit_events: Whether to emit events for this message.
        
    Returns:
        The updated state.
    """
    # Create a copy of the state
    updated_state = dict(state)
    
    # Get the current messages
    messages = state.get("messages", [])
    
    # Add the message
    updated_state["messages"] = messages + [message]
    
    # Determine if events should be emitted
    should_emit = emit_events if emit_events is not None else state.get("emit_events", True)
    
    if should_emit:
        # Determine the sender based on message type
        sender = "user"
        if isinstance(message, AIMessage):
            sender = "agent"
        elif isinstance(message, SystemMessage):
            sender = "system"
        elif isinstance(message, FunctionMessage) or isinstance(message, ToolMessage):
            sender = "tool"
        
        # Generate a message ID
        message_id = str(uuid.uuid4())
        
        # Store the message ID
        message_ids = dict(state.get("message_ids", {}))
        message_ids[str(len(messages))] = message_id
        updated_state["message_ids"] = message_ids
        
        # Emit message events
        emit_message_start(message_id, sender)
        emit_message_content(message_id, message.content)
        emit_message_end(message_id)
    
    return cast(UIMessageState, updated_state)
