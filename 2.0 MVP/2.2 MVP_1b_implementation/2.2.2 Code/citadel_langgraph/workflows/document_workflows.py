
"""
Document processing workflows for LangGraph.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union

from langgraph.graph import StateGraph, END

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager
from citadel_langchain.splitters import BaseSplitter, RecursiveCharacterTextSplitter
from citadel_langchain.vectorstores import BaseVectorStore
from citadel_langchain.retrievers import BaseRetriever

from citadel_langgraph.state.document_state import (
    DocumentProcessingState,
    DocumentExtractionState,
    DocumentSummarizationState,
    DocumentQAState,
    create_document_processing_state,
    create_document_extraction_state,
    create_document_summarization_state,
    create_document_qa_state,
)
from citadel_langgraph.nodes.document_nodes import (
    DocumentProcessingNode,
    InformationExtractionNode,
    SummarizationNode,
    QuestionAnsweringNode,
)
from citadel_langgraph.edges.document_edges import (
    ProcessingCompleteEdge,
    ExtractionCompleteEdge,
    SummarizationCompleteEdge,
    QACompleteEdge,
    DocumentProcessingStatusEdge,
)
from .base import BaseWorkflow


class DocumentProcessingWorkflow(BaseWorkflow[DocumentProcessingState]):
    """
    Workflow for processing documents.
    
    This workflow handles document processing, including splitting and metadata extraction.
    """
    
    def __init__(
        self,
        name: str = "document_processing",
        splitter: Optional[BaseSplitter] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the document processing workflow.
        
        Args:
            name: Name of the workflow.
            splitter: Text splitter to use.
            logger: Logger instance.
        """
        super().__init__(name, DocumentProcessingState, logger)
        self.splitter = splitter or RecursiveCharacterTextSplitter()
    
    def build(self) -> None:
        """
        Build the document processing workflow.
        """
        self.logger.info(f"Building document processing workflow '{self.name}'")
        
        # Create nodes
        process_node = DocumentProcessingNode(
            name="process_document",
            splitter=self.splitter,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(process_node)
        
        # Set entry point
        self.set_entry_point("process_document")
        
        # Add end node
        self.add_end_node("process_document")
    
    def run(
        self,
        source_content: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentProcessingState:
        """
        Run the document processing workflow.
        
        Args:
            source_content: Source document content.
            document_metadata: Optional document metadata.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_document_processing_state(
            source_content=source_content,
            document_metadata=document_metadata,
            metadata=metadata,
        )
        
        # Run the workflow
        return super().run(initial_state)


class InformationExtractionWorkflow(BaseWorkflow[DocumentExtractionState]):
    """
    Workflow for extracting information from documents.
    
    This workflow handles information extraction from documents.
    """
    
    def __init__(
        self,
        name: str = "information_extraction",
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the information extraction workflow.
        
        Args:
            name: Name of the workflow.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name, DocumentExtractionState, logger)
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
    
    def build(self) -> None:
        """
        Build the information extraction workflow.
        """
        self.logger.info(f"Building information extraction workflow '{self.name}'")
        
        # Create nodes
        extract_node = InformationExtractionNode(
            name="extract_information",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(extract_node)
        
        # Set entry point
        self.set_entry_point("extract_information")
        
        # Add end node
        self.add_end_node("extract_information")
    
    def run(
        self,
        source_content: str,
        extraction_schema: Dict[str, Any],
        document_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentExtractionState:
        """
        Run the information extraction workflow.
        
        Args:
            source_content: Source document content.
            extraction_schema: Schema defining what to extract.
            document_metadata: Optional document metadata.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_document_extraction_state(
            source_content=source_content,
            extraction_schema=extraction_schema,
            document_metadata=document_metadata,
            metadata=metadata,
        )
        
        # Run the workflow
        return super().run(initial_state)


class SummarizationWorkflow(BaseWorkflow[DocumentSummarizationState]):
    """
    Workflow for summarizing documents.
    
    This workflow handles document summarization.
    """
    
    def __init__(
        self,
        name: str = "summarization",
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the summarization workflow.
        
        Args:
            name: Name of the workflow.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name, DocumentSummarizationState, logger)
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
    
    def build(self) -> None:
        """
        Build the summarization workflow.
        """
        self.logger.info(f"Building summarization workflow '{self.name}'")
        
        # Create nodes
        summarize_node = SummarizationNode(
            name="summarize_document",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(summarize_node)
        
        # Set entry point
        self.set_entry_point("summarize_document")
        
        # Add end node
        self.add_end_node("summarize_document")
    
    def run(
        self,
        source_content: str,
        summarization_type: str = "abstractive",
        summary_length: str = "medium",
        document_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentSummarizationState:
        """
        Run the summarization workflow.
        
        Args:
            source_content: Source document content.
            summarization_type: Type of summarization.
            summary_length: Length of the summary.
            document_metadata: Optional document metadata.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_document_summarization_state(
            source_content=source_content,
            summarization_type=summarization_type,
            summary_length=summary_length,
            document_metadata=document_metadata,
            metadata=metadata,
        )
        
        # Run the workflow
        return super().run(initial_state)


class QuestionAnsweringWorkflow(BaseWorkflow[DocumentQAState]):
    """
    Workflow for answering questions about documents.
    
    This workflow handles document question answering.
    """
    
    def __init__(
        self,
        name: str = "question_answering",
        retriever: Optional[BaseRetriever] = None,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the question answering workflow.
        
        Args:
            name: Name of the workflow.
            retriever: Retriever to use for retrieving relevant chunks.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name, DocumentQAState, logger)
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.retriever = retriever
    
    def build(self) -> None:
        """
        Build the question answering workflow.
        """
        self.logger.info(f"Building question answering workflow '{self.name}'")
        
        # Create nodes
        qa_node = QuestionAnsweringNode(
            name="answer_question",
            retriever=self.retriever,
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(qa_node)
        
        # Set entry point
        self.set_entry_point("answer_question")
        
        # Add end node
        self.add_end_node("answer_question")
    
    def run(
        self,
        source_content: str,
        question: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentQAState:
        """
        Run the question answering workflow.
        
        Args:
            source_content: Source document content.
            question: Question to answer.
            document_metadata: Optional document metadata.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_document_qa_state(
            source_content=source_content,
            question=question,
            document_metadata=document_metadata,
            metadata=metadata,
        )
        
        # Run the workflow
        return super().run(initial_state)


class CompleteDocumentProcessingWorkflow(BaseWorkflow[DocumentProcessingState]):
    """
    Complete workflow for document processing.
    
    This workflow combines document processing, information extraction, and summarization.
    """
    
    def __init__(
        self,
        name: str = "complete_document_processing",
        splitter: Optional[BaseSplitter] = None,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        extraction_schema: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the complete document processing workflow.
        
        Args:
            name: Name of the workflow.
            splitter: Text splitter to use.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            extraction_schema: Schema defining what to extract.
            logger: Logger instance.
        """
        super().__init__(name, DocumentProcessingState, logger)
        
        self.splitter = splitter or RecursiveCharacterTextSplitter()
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.extraction_schema = extraction_schema or {
            "title": "The title of the document",
            "author": "The author of the document",
            "date": "The date of the document",
            "summary": "A brief summary of the document",
            "keywords": "Keywords or key phrases from the document",
        }
    
    def build(self) -> None:
        """
        Build the complete document processing workflow.
        """
        self.logger.info(f"Building complete document processing workflow '{self.name}'")
        
        # Create nodes
        process_node = DocumentProcessingNode(
            name="process_document",
            splitter=self.splitter,
            logger=self.logger,
        )
        
        extract_node = InformationExtractionNode(
            name="extract_information",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            logger=self.logger,
        )
        
        summarize_node = SummarizationNode(
            name="summarize_document",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(process_node)
        self.add_node(extract_node)
        self.add_node(summarize_node)
        
        # Add edges
        processing_complete_edge = ProcessingCompleteEdge(
            name="processing_complete",
            complete_node="extract_information",
            incomplete_node="process_document",  # Retry processing if incomplete
        )
        
        self.add_edge("process_document", processing_complete_edge)
        self.add_edge("extract_information", "summarize_document")
        
        # Set entry point
        self.set_entry_point("process_document")
        
        # Add end node
        self.add_end_node("summarize_document")
    
    def run(
        self,
        source_content: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentProcessingState:
        """
        Run the complete document processing workflow.
        
        Args:
            source_content: Source document content.
            document_metadata: Optional document metadata.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_document_processing_state(
            source_content=source_content,
            document_metadata=document_metadata,
            metadata=metadata,
        )
        
        # Add extraction schema to metadata
        if "metadata" not in initial_state:
            initial_state["metadata"] = {}
        initial_state["metadata"]["extraction_schema"] = self.extraction_schema
        
        # Run the workflow
        return super().run(initial_state)
