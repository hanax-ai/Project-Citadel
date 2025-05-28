
"""
Base vector store for Project Citadel LangChain integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway


class CitadelEmbeddings(Embeddings):
    """
    Embeddings class that uses Citadel's Ollama Gateway for generating embeddings.
    """
    
    def __init__(
        self,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the embeddings.
        
        Args:
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use for embeddings.
            logger: Logger instance.
        """
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.model_name = model_name
        self.logger = logger or get_logger("citadel.langchain.embeddings")
    
    async def _aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents asynchronously.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embeddings.
        """
        embeddings = []
        
        for text in texts:
            try:
                result = await self.ollama_gateway.generate_embeddings(
                    input_text=text,
                    model=self.model_name
                )
                
                embeddings.append(result["embedding"])
                
            except Exception as e:
                self.logger.error(f"Error generating embeddings: {str(e)}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 1536)  # Common embedding size
        
        return embeddings
    
    async def _aembed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query asynchronously.
        
        Args:
            text: Query text to generate embeddings for.
            
        Returns:
            Query embedding.
        """
        try:
            result = await self.ollama_gateway.generate_embeddings(
                input_text=text,
                model=self.model_name
            )
            
            return result["embedding"]
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Common embedding size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embeddings.
        """
        import asyncio
        
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._aembed_documents(texts))
        finally:
            loop.close()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query.
        
        Args:
            text: Query text to generate embeddings for.
            
        Returns:
            Query embedding.
        """
        import asyncio
        
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._aembed_query(text))
        finally:
            loop.close()


class BaseVectorStore(ABC):
    """Base class for all vector stores."""
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            embeddings: Embeddings instance to use. If None, CitadelEmbeddings will be used.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use for embeddings.
            logger: Logger instance.
        """
        self.logger = logger or get_logger(f"citadel.langchain.vectorstores.{self.__class__.__name__.lower()}")
        
        # Use provided embeddings or create CitadelEmbeddings
        self.embeddings = embeddings or CitadelEmbeddings(
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            logger=self.logger
        )
    
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add.
            metadatas: List of metadata dictionaries.
            **kwargs: Additional parameters.
            
        Returns:
            List of IDs of the added texts.
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
            **kwargs: Additional parameters.
            
        Returns:
            List of IDs of the added documents.
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Additional parameters.
            
        Returns:
            List of similar documents.
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Search for documents similar to the query and return with scores.
        
        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Additional parameters.
            
        Returns:
            List of tuples of (document, score).
        """
        pass
    
    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save to.
            **kwargs: Additional parameters.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "BaseVectorStore":
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load from.
            **kwargs: Additional parameters.
            
        Returns:
            Loaded vector store.
        """
        pass
