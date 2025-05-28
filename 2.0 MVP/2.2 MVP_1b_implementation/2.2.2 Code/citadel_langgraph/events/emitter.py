
"""
Event emitter for LangGraph workflows.

This module provides event emission capabilities for LangGraph workflows,
enabling real-time monitoring and visualization of workflow execution.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Generic, cast

from citadel_frontend.protocol.ag_ui import (
    BaseEvent, EventType, 
    StateSnapshotEvent, StateDeltaEvent,
    TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent,
    ToolCallStartEvent, ToolCallArgsEvent, ToolCallEndEvent
)
from citadel_frontend.protocol.events import EventEmitter, QueueEventEmitter, EventManager

# Global event manager instance
_event_manager: Optional[EventManager] = None

# Global event emitter instance
_event_emitter: Optional[EventEmitter] = None


def get_event_manager() -> EventManager:
    """
    Get the global event manager instance.
    
    Returns:
        EventManager: The global event manager instance
    """
    global _event_manager
    if _event_manager is None:
        _event_manager = EventManager()
    return _event_manager


def get_event_emitter() -> EventEmitter:
    """
    Get the global event emitter instance.
    
    Returns:
        EventEmitter: The global event emitter instance
    """
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = get_event_manager().create_emitter()
    return _event_emitter


def emit_event(event_type: Union[EventType, str], payload: Dict[str, Any]) -> None:
    """
    Emit an event with the specified type and payload.
    
    This function provides a synchronous interface to the asynchronous event emission,
    handling the event loop management internally.
    
    Args:
        event_type: Type of the event to emit
        payload: Payload of the event
    """
    # Convert string event type to EventType enum if needed
    if isinstance(event_type, str):
        event_type = EventType(event_type)
    
    # Create the event
    event = BaseEvent(type=event_type, payload=payload)
    
    # Get the event emitter
    emitter = get_event_emitter()
    
    # Emit the event asynchronously
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Create a task to emit the event
        asyncio.create_task(emitter.emit(event))
    else:
        # Run the event emission in the loop
        loop.run_until_complete(emitter.emit(event))


def emit_state_snapshot(state: Dict[str, Any]) -> None:
    """
    Emit a state snapshot event.
    
    Args:
        state: The complete state to emit
    """
    event = StateSnapshotEvent(state=state)
    
    # Get the event emitter
    emitter = get_event_emitter()
    
    # Emit the event asynchronously
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Create a task to emit the event
        asyncio.create_task(emitter.emit(event))
    else:
        # Run the event emission in the loop
        loop.run_until_complete(emitter.emit(event))


def emit_node_start(node_name: str, state: Dict[str, Any]) -> None:
    """
    Emit a node start event.
    
    Args:
        node_name: Name of the node that started execution
        state: Current state at node start
    """
    emit_event(EventType.THINKING_START, {
        "nodeId": node_name,
        "state": state
    })


def emit_node_end(node_name: str, state: Dict[str, Any]) -> None:
    """
    Emit a node end event.
    
    Args:
        node_name: Name of the node that finished execution
        state: Updated state after node execution
    """
    emit_event(EventType.THINKING_END, {
        "nodeId": node_name,
        "state": state
    })


def emit_tool_call_start(tool_name: str, tool_call_id: Optional[str] = None) -> str:
    """
    Emit a tool call start event.
    
    Args:
        tool_name: Name of the tool being called
        tool_call_id: Optional ID for the tool call
        
    Returns:
        str: The tool call ID
    """
    # Generate a tool call ID if not provided
    if tool_call_id is None:
        tool_call_id = str(uuid.uuid4())
    
    # Emit the event
    emit_event(EventType.TOOL_CALL_START, {
        "toolCallId": tool_call_id,
        "tool": tool_name
    })
    
    return tool_call_id


def emit_tool_call_args(tool_call_id: str, args: Dict[str, Any]) -> None:
    """
    Emit a tool call arguments event.
    
    Args:
        tool_call_id: ID of the tool call
        args: Arguments for the tool call
    """
    emit_event(EventType.TOOL_CALL_ARGS, {
        "toolCallId": tool_call_id,
        "args": args
    })


def emit_tool_call_end(tool_call_id: str, result: Any) -> None:
    """
    Emit a tool call end event.
    
    Args:
        tool_call_id: ID of the tool call
        result: Result of the tool call
    """
    # Convert result to dict if it's not already
    if not isinstance(result, dict):
        result = {"result": str(result)}
    
    emit_event(EventType.TOOL_CALL_END, {
        "toolCallId": tool_call_id,
        "result": result
    })


def emit_message_start(message_id: Optional[str] = None, sender: str = "agent") -> str:
    """
    Emit a message start event.
    
    Args:
        message_id: Optional ID for the message
        sender: Sender of the message
        
    Returns:
        str: The message ID
    """
    # Generate a message ID if not provided
    if message_id is None:
        message_id = str(uuid.uuid4())
    
    # Emit the event
    emit_event(EventType.TEXT_MESSAGE_START, {
        "messageId": message_id,
        "sender": sender
    })
    
    return message_id


def emit_message_content(message_id: str, content: str) -> None:
    """
    Emit a message content event.
    
    Args:
        message_id: ID of the message
        content: Content of the message
    """
    emit_event(EventType.TEXT_MESSAGE_CONTENT, {
        "messageId": message_id,
        "content": content
    })


def emit_message_end(message_id: str) -> None:
    """
    Emit a message end event.
    
    Args:
        message_id: ID of the message
    """
    emit_event(EventType.TEXT_MESSAGE_END, {
        "messageId": message_id
    })


def start_run() -> str:
    """
    Start a new run and emit a run started event.
    
    Returns:
        str: The run ID
    """
    # Get the event manager
    manager = get_event_manager()
    
    # Start a new run
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Create a task to start the run
        asyncio.create_task(manager.start_run())
    else:
        # Run the run start in the loop
        loop.run_until_complete(manager.start_run())
    
    return manager.run_id


def end_run() -> None:
    """
    End the current run and emit a run finished event.
    """
    # Get the event manager
    manager = get_event_manager()
    
    # End the current run
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Create a task to end the run
        asyncio.create_task(manager.end_run())
    else:
        # Run the run end in the loop
        loop.run_until_complete(manager.end_run())
