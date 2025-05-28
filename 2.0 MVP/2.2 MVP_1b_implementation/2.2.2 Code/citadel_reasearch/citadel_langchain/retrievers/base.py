
"""
Base retriever for Project Citadel LangChain integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from pydantic import ConfigDict

from citadel_core.logging import get_logger


class BaseRetriever(LangChainBaseRetriever, ABC):
    """Base class for all retrievers."""
    
    # Allow arbitrary types and extra attributes
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    def __init__(self, **kwargs):
        """
        Initialize the retriever.
        """
        # Set logger outside of Pydantic model attributes
        self._logger = kwargs.get("logger") or get_logger(f"citadel.langchain.retrievers.{self.__class__.__name__.lower()}")
    
    @property
    def logger(self):
        """Get the logger."""
        return self._logger
    
    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: Query text.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of relevant documents.
        """
        pass
    
    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get documents relevant to the query asynchronously.
        
        Args:
            query: Query text.
            **kwargs: Additional retrieval parameters.
            
        Returns:
            List of relevant documents.
        """
        # Default implementation calls the synchronous method
        # Subclasses should override this with a true async implementation if needed
        return self.get_relevant_documents(query, **kwargs)
