
"""
Document processing edges for LangGraph workflows.
"""

from typing import Any, Dict, List, Optional, Callable

from citadel_langgraph.state.document_state import (
    DocumentProcessingState,
    DocumentExtractionState,
    DocumentSummarizationState,
    DocumentQAState,
)
from .base import BaseEdge, ConditionalEdge, StatusBasedEdge, ErrorHandlingEdge


class ProcessingCompleteEdge(ConditionalEdge[DocumentProcessingState]):
    """
    Edge that routes based on whether document processing is complete.
    
    This class implements processing completion routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        complete_node: str,
        incomplete_node: str,
    ):
        """
        Initialize the processing complete edge.
        
        Args:
            name: Name of the edge.
            complete_node: Node to route to if processing is complete.
            incomplete_node: Node to route to if processing is incomplete.
        """
        super().__init__(
            name=name,
            condition=lambda state: state.get("processing_status") == "completed",
            true_node=complete_node,
            false_node=incomplete_node,
        )


class ExtractionCompleteEdge(ConditionalEdge[DocumentExtractionState]):
    """
    Edge that routes based on whether information extraction is complete.
    
    This class implements extraction completion routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        complete_node: str,
        incomplete_node: str,
    ):
        """
        Initialize the extraction complete edge.
        
        Args:
            name: Name of the edge.
            complete_node: Node to route to if extraction is complete.
            incomplete_node: Node to route to if extraction is incomplete.
        """
        super().__init__(
            name=name,
            condition=lambda state: bool(state.get("extracted_data")),
            true_node=complete_node,
            false_node=incomplete_node,
        )


class SummarizationCompleteEdge(ConditionalEdge[DocumentSummarizationState]):
    """
    Edge that routes based on whether document summarization is complete.
    
    This class implements summarization completion routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        complete_node: str,
        incomplete_node: str,
    ):
        """
        Initialize the summarization complete edge.
        
        Args:
            name: Name of the edge.
            complete_node: Node to route to if summarization is complete.
            incomplete_node: Node to route to if summarization is incomplete.
        """
        super().__init__(
            name=name,
            condition=lambda state: state.get("summary") is not None,
            true_node=complete_node,
            false_node=incomplete_node,
        )


class QACompleteEdge(ConditionalEdge[DocumentQAState]):
    """
    Edge that routes based on whether question answering is complete.
    
    This class implements QA completion routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        complete_node: str,
        incomplete_node: str,
    ):
        """
        Initialize the QA complete edge.
        
        Args:
            name: Name of the edge.
            complete_node: Node to route to if QA is complete.
            incomplete_node: Node to route to if QA is incomplete.
        """
        super().__init__(
            name=name,
            condition=lambda state: state.get("answer") is not None,
            true_node=complete_node,
            false_node=incomplete_node,
        )


class DocumentProcessingStatusEdge(StatusBasedEdge[DocumentProcessingState]):
    """
    Edge that routes based on the document processing status.
    
    This class implements processing status routing in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        status_routes: Dict[str, str],
        default_route: Optional[str] = None,
    ):
        """
        Initialize the document processing status edge.
        
        Args:
            name: Name of the edge.
            status_routes: Dictionary mapping processing status values to node names.
            default_route: Default node to route to if status is not found.
        """
        super().__init__(
            name=name,
            status_routes=status_routes,
            default_route=default_route,
        )
    
    def __call__(self, state: DocumentProcessingState) -> str:
        """
        Determine the next node based on the processing status.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Name of the next node.
        """
        processing_status = state.get("processing_status", "")
        return self.status_routes.get(processing_status, self.default_route)
