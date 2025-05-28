
"""
Base document loader for Project Citadel LangChain integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from citadel_core.logging import get_logger


class BaseLoader(ABC):
    """Base class for all document loaders."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the loader.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger or get_logger(f"citadel.langchain.loaders.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    def load(self, source: Any, **kwargs) -> List[Document]:
        """
        Load documents from the source.
        
        Args:
            source: Source to load documents from.
            **kwargs: Additional loading parameters.
            
        Returns:
            List of loaded documents.
        """
        pass
    
    def _create_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Create a LangChain Document from text and metadata.
        
        Args:
            text: Document text.
            metadata: Document metadata.
            
        Returns:
            LangChain Document.
        """
        return Document(
            page_content=text,
            metadata=metadata or {}
        )
