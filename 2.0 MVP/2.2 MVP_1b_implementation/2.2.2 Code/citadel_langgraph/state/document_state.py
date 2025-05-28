
"""
Specialized state classes for document processing workflows.
"""

from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union
from datetime import datetime

from langchain_core.documents import Document

from .base import BaseState, DocumentState, create_document_state


class DocumentProcessingState(DocumentState):
    """
    State class for document processing workflows.
    
    This class extends DocumentState to include processing-specific fields.
    """
    
    # Processing status
    processing_status: str
    
    # Processing steps completed
    completed_steps: List[str]
    
    # Processing results
    processing_results: Dict[str, Any]


def create_document_processing_state(
    source_content: str,
    document_metadata: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentProcessingState:
    """
    Create a new document processing state with default values.
    
    Args:
        source_content: Source document content.
        document_metadata: Optional document metadata.
        metadata: Optional workflow metadata.
        
    Returns:
        A new DocumentProcessingState instance.
    """
    doc_state = create_document_state(source_content, document_metadata, metadata)
    return DocumentProcessingState(
        **doc_state,
        processing_status="pending",
        completed_steps=[],
        processing_results={},
    )


class DocumentExtractionState(DocumentState):
    """
    State class for information extraction workflows.
    
    This class extends DocumentState to include extraction-specific fields.
    """
    
    # Extraction schema
    extraction_schema: Dict[str, Any]
    
    # Extracted data
    extracted_data: Dict[str, Any]
    
    # Extraction confidence scores
    confidence_scores: Dict[str, float]


def create_document_extraction_state(
    source_content: str,
    extraction_schema: Dict[str, Any],
    document_metadata: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentExtractionState:
    """
    Create a new document extraction state with default values.
    
    Args:
        source_content: Source document content.
        extraction_schema: Schema defining what to extract.
        document_metadata: Optional document metadata.
        metadata: Optional workflow metadata.
        
    Returns:
        A new DocumentExtractionState instance.
    """
    doc_state = create_document_state(source_content, document_metadata, metadata)
    return DocumentExtractionState(
        **doc_state,
        extraction_schema=extraction_schema,
        extracted_data={},
        confidence_scores={},
    )


class DocumentSummarizationState(DocumentState):
    """
    State class for document summarization workflows.
    
    This class extends DocumentState to include summarization-specific fields.
    """
    
    # Summarization type (e.g., extractive, abstractive)
    summarization_type: str
    
    # Summary length (e.g., short, medium, long)
    summary_length: str
    
    # Generated summary
    summary: Optional[str]
    
    # Key points extracted
    key_points: List[str]


def create_document_summarization_state(
    source_content: str,
    summarization_type: str = "abstractive",
    summary_length: str = "medium",
    document_metadata: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentSummarizationState:
    """
    Create a new document summarization state with default values.
    
    Args:
        source_content: Source document content.
        summarization_type: Type of summarization.
        summary_length: Length of the summary.
        document_metadata: Optional document metadata.
        metadata: Optional workflow metadata.
        
    Returns:
        A new DocumentSummarizationState instance.
    """
    doc_state = create_document_state(source_content, document_metadata, metadata)
    return DocumentSummarizationState(
        **doc_state,
        summarization_type=summarization_type,
        summary_length=summary_length,
        summary=None,
        key_points=[],
    )


class DocumentQAState(DocumentState):
    """
    State class for document question answering workflows.
    
    This class extends DocumentState to include QA-specific fields.
    """
    
    # Question asked
    question: str
    
    # Answer generated
    answer: Optional[str]
    
    # Relevant document chunks
    relevant_chunks: List[Dict[str, Any]]
    
    # Answer sources
    sources: List[Dict[str, Any]]
    
    # Confidence score
    confidence: Optional[float]


def create_document_qa_state(
    source_content: str,
    question: str,
    document_metadata: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DocumentQAState:
    """
    Create a new document QA state with default values.
    
    Args:
        source_content: Source document content.
        question: Question to answer.
        document_metadata: Optional document metadata.
        metadata: Optional workflow metadata.
        
    Returns:
        A new DocumentQAState instance.
    """
    doc_state = create_document_state(source_content, document_metadata, metadata)
    return DocumentQAState(
        **doc_state,
        question=question,
        answer=None,
        relevant_chunks=[],
        sources=[],
        confidence=None,
    )
