
"""
Base text splitter for Project Citadel LangChain integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from citadel_core.logging import get_logger


class BaseSplitter(ABC):
    """Base class for all text splitters."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the splitter.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger or get_logger(f"citadel.langchain.splitters.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    def split_text(self, text: str, **kwargs) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split.
            **kwargs: Additional splitting parameters.
            
        Returns:
            List of text chunks.
        """
        pass
    
    def split_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: Documents to split.
            **kwargs: Additional splitting parameters.
            
        Returns:
            List of split documents.
        """
        result = []
        
        for doc in documents:
            try:
                # Split the text
                chunks = self.split_text(doc.page_content, **kwargs)
                
                # Create new documents for each chunk
                for i, chunk in enumerate(chunks):
                    # Create a copy of the metadata
                    metadata = doc.metadata.copy()
                    
                    # Add chunk information to metadata
                    metadata["chunk"] = i + 1
                    metadata["total_chunks"] = len(chunks)
                    
                    # Create a new document
                    new_doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    
                    result.append(new_doc)
                    
            except Exception as e:
                self.logger.error(f"Error splitting document: {str(e)}")
                # Include the original document if splitting fails
                result.append(doc)
        
        return result
    
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
