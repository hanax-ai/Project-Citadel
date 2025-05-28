
"""
Nodes module for Citadel LangGraph.

This module provides nodes that can be used in LangGraph workflows.
"""

from .base import BaseNode, FunctionNode
from .agent_nodes import (
    LLMNode,
    ReActAgentNode,
    ToolExecutionNode,
    AgentCoordinatorNode,
)
from .document_nodes import (
    DocumentProcessingNode,
    InformationExtractionNode as DocumentChunkingNode,  # Alias for backward compatibility
    QuestionAnsweringNode as DocumentEmbeddingNode,  # Alias for backward compatibility
)
from .reflection_node import ReflectionNode
from .planning_node import PlanningNode

__all__ = [
    "BaseNode",
    "FunctionNode",
    "LLMNode",
    "ReActAgentNode",
    "ToolExecutionNode",
    "AgentCoordinatorNode",
    "DocumentProcessingNode",
    "DocumentChunkingNode",
    "DocumentEmbeddingNode",
    "ReflectionNode",
    "PlanningNode",
]
