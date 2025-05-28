
"""
Vector store retriever for Project Citadel LangChain integration.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from pydantic import Field, ConfigDict

from citadel_core.logging import get_logger

from ..vectorstores.base import BaseVectorStore
from .base import BaseRetriever


class VectorStoreRetriever(BaseRetriever):
    """Retriever that uses a vector store for retrieval."""
    
    # Allow arbitrary types and extra attributes
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the vector store retriever.
        
        Args:
            vector_store: Vector store to use for retrieval.
            search_kwargs: Additional search parameters.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 4}
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: Query text.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of relevant documents.
        """
        # Merge search kwargs
        search_kwargs = {**self.search_kwargs, **kwargs}
        
        # Search the vector store
        return self.vector_store.similarity_search(query, **search_kwargs)
    
    def get_relevant_documents_with_scores(self, query: str, **kwargs) -> List[tuple[Document, float]]:
        """
        Get documents relevant to the query with scores.
        
        Args:
            query: Query text.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of tuples of (document, score).
        """
        # Merge search kwargs
        search_kwargs = {**self.search_kwargs, **kwargs}
        
        # Search the vector store
        return self.vector_store.similarity_search_with_score(query, **search_kwargs)
