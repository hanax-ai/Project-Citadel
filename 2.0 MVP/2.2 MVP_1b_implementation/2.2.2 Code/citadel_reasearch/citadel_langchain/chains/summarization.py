
"""
Summarization chain for Project Citadel LangChain integration.
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

from .base import BaseChain


class SummarizationChain(BaseChain):
    """Chain for summarizing documents."""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        prompt: Optional[BasePromptTemplate] = None,
        output_parser: Optional[BaseOutputParser] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the summarization chain.
        
        Args:
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
                "You are a helpful assistant that summarizes documents.\n\n"
                "Document to summarize:\n{document}\n\n"
                "Instructions: {instructions}\n\n"
                "Summary:"
            )
        
        super().__init__(
            llm_manager=llm_manager,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            prompt=prompt,
            output_parser=output_parser,
            logger=logger
        )
    
    async def _arun(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the summarization chain asynchronously.
        
        Args:
            inputs: Input values including the document and instructions.
            **kwargs: Additional parameters.
            
        Returns:
            Output values including the summary.
        """
        # Extract the document and instructions
        document = inputs.get("document")
        if not document:
            raise ValueError("Document not provided")
        
        instructions = inputs.get("instructions", "Provide a concise summary of the document.")
        
        # Generate the summary
        summary = await self._agenerate_from_prompt(
            {
                "document": document,
                "instructions": instructions
            },
            **kwargs
        )
        
        # Parse the output if needed
        parsed_summary = self._parse_output(summary)
        
        return {
            "document": document,
            "instructions": instructions,
            "summary": parsed_summary
        }
    
    async def summarize_documents(
        self,
        documents: List[Document],
        instructions: str = "Provide a concise summary of the documents.",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Summarize a list of documents.
        
        Args:
            documents: List of documents to summarize.
            instructions: Instructions for the summarization.
            **kwargs: Additional parameters.
            
        Returns:
            Output values including the summary.
        """
        if not documents:
            return {"summary": "No documents provided."}
        
        # Combine all documents
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Run the chain
        return await self._arun(
            {
                "document": combined_text,
                "instructions": instructions
            },
            **kwargs
        )
