
"""
Base state classes for LangGraph workflows.
"""

from abc import ABC
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union
from datetime import datetime
import json
import uuid

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    BaseMessage,
)
from operator import add


class BaseState(TypedDict, total=False):
    """
    Base state class for LangGraph workflows.
    
    This class defines the common state structure for all workflows.
    It can be extended for specific workflow types.
    """
    
    # Unique identifier for the workflow instance
    workflow_id: str
    
    # Timestamp when the workflow was created
    created_at: str
    
    # Timestamp when the workflow was last updated
    updated_at: str
    
    # Current status of the workflow
    status: str
    
    # Error message if any
    error: Optional[str]
    
    # Additional metadata
    metadata: Dict[str, Any]


def create_base_state(metadata: Optional[Dict[str, Any]] = None) -> BaseState:
    """
    Create a new base state with default values.
    
    Args:
        metadata: Optional metadata to include in the state.
        
    Returns:
        A new BaseState instance.
    """
    now = datetime.utcnow().isoformat()
    return BaseState(
        workflow_id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        status="initialized",
        error=None,
        metadata=metadata or {},
    )


class MessageState(BaseState):
    """
    State class for workflows that involve message exchanges.
    
    This class extends BaseState to include a list of messages.
    """
    
    # List of messages in the conversation
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage]], add]


def create_message_state(
    system_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> MessageState:
    """
    Create a new message state with default values.
    
    Args:
        system_message: Optional system message to include in the state.
        metadata: Optional metadata to include in the state.
        
    Returns:
        A new MessageState instance.
    """
    base_state = create_base_state(metadata)
    messages = []
    
    if system_message:
        messages.append(SystemMessage(content=system_message))
    
    return MessageState(
        **base_state,
        messages=messages,
    )


class DocumentState(BaseState):
    """
    State class for workflows that involve document processing.
    
    This class extends BaseState to include document-related fields.
    """
    
    # Source document content
    source_content: str
    
    # Processed document content
    processed_content: Optional[str]
    
    # Document metadata
    document_metadata: Dict[str, Any]
    
    # Extracted entities
    entities: List[Dict[str, Any]]
    
    # Document chunks
    chunks: List[Dict[str, Any]]


def create_document_state(
    source_content: str,
    document_metadata: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentState:
    """
    Create a new document state with default values.
    
    Args:
        source_content: Source document content.
        document_metadata: Optional document metadata.
        metadata: Optional workflow metadata.
        
    Returns:
        A new DocumentState instance.
    """
    base_state = create_base_state(metadata)
    return DocumentState(
        **base_state,
        source_content=source_content,
        processed_content=None,
        document_metadata=document_metadata or {},
        entities=[],
        chunks=[],
    )


class AgentState(MessageState):
    """
    State class for agent workflows.
    
    This class extends MessageState to include agent-specific fields.
    """
    
    # Current step in the agent's execution
    current_step: str
    
    # History of steps executed by the agent
    step_history: List[Dict[str, Any]]
    
    # Tools available to the agent
    tools: List[Dict[str, Any]]
    
    # Tool calls made by the agent
    tool_calls: List[Dict[str, Any]]
    
    # Tool results returned to the agent
    tool_results: List[Dict[str, Any]]


def create_agent_state(
    system_message: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Create a new agent state with default values.
    
    Args:
        system_message: Optional system message to include in the state.
        tools: Optional list of tools available to the agent.
        metadata: Optional metadata to include in the state.
        
    Returns:
        A new AgentState instance.
    """
    message_state = create_message_state(system_message, metadata)
    return AgentState(
        **message_state,
        current_step="initialize",
        step_history=[],
        tools=tools or [],
        tool_calls=[],
        tool_results=[],
    )


class StatePersistence(ABC):
    """
    Base class for state persistence mechanisms.
    
    This class defines the interface for persisting and retrieving workflow state.
    """
    
    def save_state(self, state: BaseState) -> None:
        """
        Save the state to persistent storage.
        
        Args:
            state: The state to save.
        """
        raise NotImplementedError("Subclasses must implement save_state")
    
    def load_state(self, workflow_id: str) -> BaseState:
        """
        Load the state from persistent storage.
        
        Args:
            workflow_id: The ID of the workflow to load.
            
        Returns:
            The loaded state.
        """
        raise NotImplementedError("Subclasses must implement load_state")
    
    def delete_state(self, workflow_id: str) -> None:
        """
        Delete the state from persistent storage.
        
        Args:
            workflow_id: The ID of the workflow to delete.
        """
        raise NotImplementedError("Subclasses must implement delete_state")


class FileStatePersistence(StatePersistence):
    """
    File-based state persistence mechanism.
    
    This class implements state persistence using JSON files.
    """
    
    def __init__(self, directory: str = "/tmp/citadel_states"):
        """
        Initialize the file-based state persistence.
        
        Args:
            directory: Directory to store state files.
        """
        import os
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
    
    def _get_file_path(self, workflow_id: str) -> str:
        """
        Get the file path for a workflow ID.
        
        Args:
            workflow_id: The ID of the workflow.
            
        Returns:
            The file path.
        """
        import os
        return os.path.join(self.directory, f"{workflow_id}.json")
    
    def save_state(self, state: BaseState) -> None:
        """
        Save the state to a JSON file.
        
        Args:
            state: The state to save.
        """
        workflow_id = state.get("workflow_id")
        if not workflow_id:
            raise ValueError("State must have a workflow_id")
        
        # Update the updated_at timestamp
        state["updated_at"] = datetime.utcnow().isoformat()
        
        # Convert messages to dict for serialization
        serializable_state = self._prepare_for_serialization(state)
        
        with open(self._get_file_path(workflow_id), "w") as f:
            json.dump(serializable_state, f, indent=2)
    
    def load_state(self, workflow_id: str) -> BaseState:
        """
        Load the state from a JSON file.
        
        Args:
            workflow_id: The ID of the workflow to load.
            
        Returns:
            The loaded state.
        """
        try:
            with open(self._get_file_path(workflow_id), "r") as f:
                state_dict = json.load(f)
            
            # Convert serialized messages back to message objects
            return self._prepare_from_serialization(state_dict)
        except FileNotFoundError:
            raise ValueError(f"No state found for workflow ID: {workflow_id}")
    
    def delete_state(self, workflow_id: str) -> None:
        """
        Delete the state file.
        
        Args:
            workflow_id: The ID of the workflow to delete.
        """
        import os
        try:
            os.remove(self._get_file_path(workflow_id))
        except FileNotFoundError:
            pass  # Ignore if file doesn't exist
    
    def _prepare_for_serialization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the state for serialization.
        
        Args:
            state: The state to prepare.
            
        Returns:
            The prepared state.
        """
        serializable_state = dict(state)
        
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
        
        return serializable_state
    
    def _prepare_from_serialization(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the state from serialization.
        
        Args:
            state_dict: The serialized state.
            
        Returns:
            The prepared state.
        """
        # Handle message deserialization
        if "messages" in state_dict:
            messages = []
            for msg_dict in state_dict["messages"]:
                msg_type = msg_dict["type"]
                if msg_type == "HumanMessage":
                    msg = HumanMessage(content=msg_dict["content"], additional_kwargs=msg_dict.get("additional_kwargs", {}))
                elif msg_type == "AIMessage":
                    msg = AIMessage(content=msg_dict["content"], additional_kwargs=msg_dict.get("additional_kwargs", {}))
                elif msg_type == "SystemMessage":
                    msg = SystemMessage(content=msg_dict["content"], additional_kwargs=msg_dict.get("additional_kwargs", {}))
                elif msg_type == "FunctionMessage":
                    msg = FunctionMessage(content=msg_dict["content"], additional_kwargs=msg_dict.get("additional_kwargs", {}))
                elif msg_type == "ToolMessage":
                    msg = ToolMessage(content=msg_dict["content"], additional_kwargs=msg_dict.get("additional_kwargs", {}))
                else:
                    raise ValueError(f"Unknown message type: {msg_type}")
                messages.append(msg)
            state_dict["messages"] = messages
        
        return state_dict
