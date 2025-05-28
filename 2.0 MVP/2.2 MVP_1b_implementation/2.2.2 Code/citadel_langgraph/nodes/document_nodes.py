
"""
Document processing nodes for LangGraph workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager
from citadel_langchain.splitters import BaseSplitter, RecursiveSplitter
from citadel_langchain.vectorstores import BaseVectorStore
from citadel_langchain.retrievers import BaseRetriever

from citadel_langgraph.state.base import BaseState
from citadel_langgraph.state.document_state import (
    DocumentState,
    DocumentProcessingState,
    DocumentExtractionState,
    DocumentSummarizationState,
    DocumentQAState,
)
from .base import BaseNode, FunctionNode


class DocumentProcessingNode(BaseNode[DocumentProcessingState]):
    """
    Node for processing documents.
    
    This class implements document processing logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        splitter: Optional[BaseSplitter] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the document processing node.
        
        Args:
            name: Name of the node.
            splitter: Text splitter to use.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        self.splitter = splitter or RecursiveSplitter()
    
    def __call__(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """
        Process the document.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Processing document in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        updated_state["processing_status"] = "processing"
        
        try:
            # Process the document
            source_content = state["source_content"]
            document_metadata = state["document_metadata"]
            
            # Create a Document object
            doc = Document(page_content=source_content, metadata=document_metadata)
            
            # Split the document
            chunks = self.splitter.split_documents([doc])
            
            # Update state with chunks
            updated_state["chunks"] = [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
            
            # Mark processing as complete
            updated_state["processing_status"] = "completed"
            updated_state["completed_steps"] = state.get("completed_steps", []) + ["document_processing"]
            updated_state["processed_content"] = source_content
            
            self.logger.info(f"Document processing completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            updated_state["processing_status"] = "error"
            updated_state["error"] = str(e)
        
        return updated_state


class InformationExtractionNode(BaseNode[DocumentExtractionState]):
    """
    Node for extracting information from documents.
    
    This class implements information extraction logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the information extraction node.
        
        Args:
            name: Name of the node.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
    
    def __call__(self, state: DocumentExtractionState) -> DocumentExtractionState:
        """
        Extract information from the document.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Extracting information from document in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get document content and extraction schema
            source_content = state["source_content"]
            extraction_schema = state["extraction_schema"]
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(source_content, extraction_schema)
            
            # Extract information using LLM
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.llm_manager.generate(
                        prompt=prompt,
                        model_name=self.model_name,
                    )
                )
            finally:
                loop.close()
            
            # Parse the result
            extracted_data = self._parse_extraction_result(result.text)
            
            # Update state with extracted data
            updated_state["extracted_data"] = extracted_data
            updated_state["confidence_scores"] = {
                key: 0.9  # Placeholder confidence score
                for key in extracted_data
            }
            
            self.logger.info(f"Information extraction completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error extracting information: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def _create_extraction_prompt(self, content: str, schema: Dict[str, Any]) -> str:
        """
        Create a prompt for information extraction.
        
        Args:
            content: Document content.
            schema: Extraction schema.
            
        Returns:
            Extraction prompt.
        """
        # Convert schema to a string representation
        schema_str = "\n".join([f"- {key}: {value}" for key, value in schema.items()])
        
        return (
            "Extract the following information from the document:\n\n"
            f"{schema_str}\n\n"
            "Document content:\n"
            f"{content}\n\n"
            "Provide the extracted information in JSON format."
        )
    
    def _parse_extraction_result(self, result: str) -> Dict[str, Any]:
        """
        Parse the extraction result.
        
        Args:
            result: Extraction result from LLM.
            
        Returns:
            Parsed extraction result.
        """
        # Simple parsing for now - in a real implementation, this would be more robust
        import json
        try:
            # Try to find JSON in the result
            start_idx = result.find("{")
            end_idx = result.rfind("}")
            
            if start_idx >= 0 and end_idx >= 0:
                json_str = result[start_idx:end_idx + 1]
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in extraction result")
                return {}
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse extraction result as JSON")
            return {}


class SummarizationNode(BaseNode[DocumentSummarizationState]):
    """
    Node for summarizing documents.
    
    This class implements document summarization logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the summarization node.
        
        Args:
            name: Name of the node.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
    
    def __call__(self, state: DocumentSummarizationState) -> DocumentSummarizationState:
        """
        Summarize the document.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Summarizing document in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get document content and summarization parameters
            source_content = state["source_content"]
            summarization_type = state["summarization_type"]
            summary_length = state["summary_length"]
            
            # Create summarization prompt
            prompt = self._create_summarization_prompt(
                source_content, summarization_type, summary_length
            )
            
            # Generate summary using LLM
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.llm_manager.generate(
                        prompt=prompt,
                        model_name=self.model_name,
                    )
                )
            finally:
                loop.close()
            
            # Parse the result
            summary, key_points = self._parse_summarization_result(result.text)
            
            # Update state with summary and key points
            updated_state["summary"] = summary
            updated_state["key_points"] = key_points
            
            self.logger.info(f"Document summarization completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error summarizing document: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def _create_summarization_prompt(
        self, content: str, summarization_type: str, summary_length: str
    ) -> str:
        """
        Create a prompt for document summarization.
        
        Args:
            content: Document content.
            summarization_type: Type of summarization.
            summary_length: Length of the summary.
            
        Returns:
            Summarization prompt.
        """
        # Define length in words based on summary_length
        length_map = {
            "short": "100-150 words",
            "medium": "250-300 words",
            "long": "500-600 words",
        }
        length_str = length_map.get(summary_length, "250-300 words")
        
        # Define summarization type instructions
        type_instructions = {
            "extractive": "Extract the most important sentences from the document.",
            "abstractive": "Create a coherent summary in your own words.",
        }
        type_str = type_instructions.get(
            summarization_type, "Create a coherent summary in your own words."
        )
        
        return (
            f"Please summarize the following document. {type_str} "
            f"The summary should be approximately {length_str}.\n\n"
            "After the summary, please list 3-5 key points from the document.\n\n"
            "Document content:\n"
            f"{content}\n\n"
            "Summary:"
        )
    
    def _parse_summarization_result(self, result: str) -> tuple[str, List[str]]:
        """
        Parse the summarization result.
        
        Args:
            result: Summarization result from LLM.
            
        Returns:
            Tuple of (summary, key_points).
        """
        # Simple parsing - in a real implementation, this would be more robust
        lines = result.strip().split("\n")
        
        # Extract summary (everything before "Key Points" or similar)
        summary_lines = []
        key_points_start = -1
        
        for i, line in enumerate(lines):
            if any(kp in line.lower() for kp in ["key point", "key points", "main point", "main points"]):
                key_points_start = i
                break
            summary_lines.append(line)
        
        summary = "\n".join(summary_lines).strip()
        
        # Extract key points
        key_points = []
        if key_points_start >= 0:
            for line in lines[key_points_start + 1:]:
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                    # Remove leading markers (-, *, 1., etc.)
                    point = line.lstrip("-*0123456789. \t")
                    key_points.append(point)
        
        return summary, key_points


class QuestionAnsweringNode(BaseNode[DocumentQAState]):
    """
    Node for answering questions about documents.
    
    This class implements document QA logic in a LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        retriever: Optional[BaseRetriever] = None,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the question answering node.
        
        Args:
            name: Name of the node.
            retriever: Retriever to use for retrieving relevant chunks.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            logger: Logger instance.
        """
        super().__init__(name)
        self.logger = logger or get_logger(f"citadel.langgraph.nodes.{name}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.retriever = retriever
    
    def __call__(self, state: DocumentQAState) -> DocumentQAState:
        """
        Answer a question about the document.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated workflow state.
        """
        self.logger.info(f"Answering question in workflow {state['workflow_id']}")
        
        # Update state
        updated_state = dict(state)
        
        try:
            # Get document content and question
            source_content = state["source_content"]
            question = state["question"]
            
            # If we have chunks and a retriever, use them to get relevant chunks
            relevant_chunks = []
            if state.get("chunks") and self.retriever:
                # Convert chunks to Documents
                docs = [
                    Document(
                        page_content=chunk["content"],
                        metadata=chunk.get("metadata", {})
                    )
                    for chunk in state.get("chunks", [])
                ]
                
                # Get relevant chunks
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    relevant_docs = loop.run_until_complete(
                        self.retriever.aget_relevant_documents(question)
                    )
                finally:
                    loop.close()
                
                # Convert relevant docs to chunks
                relevant_chunks = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in relevant_docs
                ]
            else:
                # If no chunks or retriever, use the whole document
                relevant_chunks = [{
                    "content": source_content,
                    "metadata": state.get("document_metadata", {}),
                }]
            
            # Create QA prompt
            prompt = self._create_qa_prompt(question, relevant_chunks)
            
            # Generate answer using LLM
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.llm_manager.generate(
                        prompt=prompt,
                        model_name=self.model_name,
                    )
                )
            finally:
                loop.close()
            
            # Update state with answer and relevant chunks
            updated_state["answer"] = result.text
            updated_state["relevant_chunks"] = relevant_chunks
            updated_state["sources"] = [
                {
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                }
                for chunk in relevant_chunks
            ]
            updated_state["confidence"] = 0.9  # Placeholder confidence score
            
            self.logger.info(f"Question answering completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            updated_state["error"] = str(e)
        
        return updated_state
    
    def _create_qa_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for question answering.
        
        Args:
            question: Question to answer.
            chunks: Relevant document chunks.
            
        Returns:
            QA prompt.
        """
        # Format chunks
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append(f"Chunk {i+1}:\n{chunk['content']}")
        
        chunks_text = "\n\n".join(formatted_chunks)
        
        return (
            "Answer the following question based on the provided document chunks. "
            "If the answer is not in the chunks, say that you don't know.\n\n"
            f"Question: {question}\n\n"
            "Document chunks:\n"
            f"{chunks_text}\n\n"
            "Answer:"
        )
