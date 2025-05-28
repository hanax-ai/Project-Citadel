
"""
Question answering chain for Project Citadel LangChain integration.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from ..retrievers.base import BaseRetriever
from .base import BaseChain


class QAChain(BaseChain):
    """Chain for question answering using a retriever and LLM."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        prompt: Optional[BasePromptTemplate] = None,
        output_parser: Optional[BaseOutputParser] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the QA chain.
        
        Args:
            retriever: Retriever to use for retrieving documents.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            prompt: Prompt template to use.
            output_parser: Output parser to use.
            logger: Logger instance.
        """
        # Default prompt if none provided
        if prompt is None:
            prompt = PromptTemplate.from_template(
                "You are a helpful assistant that answers questions based on the provided context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        
        super().__init__(
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            prompt=prompt,
            output_parser=output_parser,
            logger=logger
        )
        
        self.retriever = retriever
    
    async def _arun(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the QA chain asynchronously.
        
        Args:
            inputs: Input values including the question.
            **kwargs: Additional parameters.
            
        Returns:
            Output values including the answer.
        """
        # Extract the question
        question = inputs.get("question")
        if not question:
            raise ValueError("Question not provided")
        
        # Retrieve relevant documents
        docs = await self.retriever.aget_relevant_documents(question, **kwargs)
        
        # Format the context
        context = self._format_documents(docs)
        
        # Generate the answer
        answer = await self._agenerate_from_prompt(
            {
                "context": context,
                "question": question
            },
            **kwargs
        )
        
        # Parse the output if needed
        parsed_answer = self._parse_output(answer)
        
        return {
            "question": question,
            "answer": parsed_answer,
            "context": context,
            "source_documents": docs
        }
    
    def _format_documents(self, docs: List[Document]) -> str:
        """
        Format documents for inclusion in the prompt.
        
        Args:
            docs: List of documents.
            
        Returns:
            Formatted context string.
        """
        if not docs:
            return "No relevant information found."
        
        # Format each document with its metadata
        formatted_docs = []
        for i, doc in enumerate(docs):
            # Extract source information
            source = doc.metadata.get("source", f"Document {i+1}")
            
            # Format the document
            formatted_doc = f"Document {i+1} (Source: {source}):\n{doc.page_content}"
            formatted_docs.append(formatted_doc)
        
        # Join all formatted documents
        return "\n\n".join(formatted_docs)
