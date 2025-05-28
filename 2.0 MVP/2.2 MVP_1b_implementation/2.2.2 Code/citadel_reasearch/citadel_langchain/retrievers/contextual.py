
"""
Contextual retriever for Project Citadel LangChain integration.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from pydantic import Field, ConfigDict

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from .base import BaseRetriever
from .vector import VectorStoreRetriever


class ContextualRetriever(BaseRetriever):
    """
    Retriever that uses an LLM to rewrite the query for better retrieval.
    """
    
    # Allow arbitrary types and extra attributes
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        query_transformation_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the contextual retriever.
        
        Args:
            base_retriever: Base retriever to use for retrieval.
            llm_manager: LLM manager to use for query transformation.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use for query transformation.
            query_transformation_prompt: Prompt to use for query transformation.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        self.base_retriever = base_retriever
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        
        # Default query transformation prompt
        self.query_transformation_prompt = query_transformation_prompt or (
            "You are an AI assistant helping to improve search queries. "
            "Given the original query, rewrite it to be more effective for retrieving relevant information. "
            "Focus on extracting key concepts and using synonyms or related terms that might appear in relevant documents. "
            "Original query: {query}\n\n"
            "Improved query:"
        )
    
    async def _transform_query(self, query: str) -> str:
        """
        Transform the query using the LLM.
        
        Args:
            query: Original query.
            
        Returns:
            Transformed query.
        """
        try:
            # Format the prompt
            prompt = self.query_transformation_prompt.format(query=query)
            
            # Generate the transformed query
            result = await self.llm_manager.generate(
                prompt=prompt,
                model_name=self.model_name,
                stream=False
            )
            
            transformed_query = result.text.strip()
            
            self.logger.info(f"Transformed query: '{query}' -> '{transformed_query}'")
            
            return transformed_query
            
        except Exception as e:
            self.logger.error(f"Error transforming query: {str(e)}")
            # Return the original query if transformation fails
            return query
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: Query text.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of relevant documents.
        """
        import asyncio
        
        # Transform the query
        loop = asyncio.new_event_loop()
        try:
            transformed_query = loop.run_until_complete(self._transform_query(query))
        finally:
            loop.close()
        
        # Use the transformed query for retrieval
        use_original = kwargs.pop("use_original", False)
        if use_original:
            # Use both queries and combine results
            original_docs = self.base_retriever.get_relevant_documents(query, **kwargs)
            transformed_docs = self.base_retriever.get_relevant_documents(transformed_query, **kwargs)
            
            # Combine and deduplicate
            seen_contents = set()
            combined_docs = []
            
            for doc in original_docs + transformed_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    combined_docs.append(doc)
            
            return combined_docs[:kwargs.get("k", 4)]
        else:
            # Use only the transformed query
            return self.base_retriever.get_relevant_documents(transformed_query, **kwargs)
    
    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get documents relevant to the query asynchronously.
        
        Args:
            query: Query text.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of relevant documents.
        """
        # Transform the query
        transformed_query = await self._transform_query(query)
        
        # Use the transformed query for retrieval
        use_original = kwargs.pop("use_original", False)
        if use_original:
            # Use both queries and combine results
            original_docs = await self.base_retriever.aget_relevant_documents(query, **kwargs)
            transformed_docs = await self.base_retriever.aget_relevant_documents(transformed_query, **kwargs)
            
            # Combine and deduplicate
            seen_contents = set()
            combined_docs = []
            
            for doc in original_docs + transformed_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    combined_docs.append(doc)
            
            return combined_docs[:kwargs.get("k", 4)]
        else:
            # Use only the transformed query
            return await self.base_retriever.aget_relevant_documents(transformed_query, **kwargs)
