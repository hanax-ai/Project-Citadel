
"""
In-memory vector store for Project Citadel LangChain integration.
"""

import logging
import pickle
import os
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway

from .base import BaseVectorStore, CitadelEmbeddings


class InMemoryVectorStore(BaseVectorStore):
    """
    Simple in-memory vector store for storing and retrieving embeddings.
    """
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the in-memory vector store.
        
        Args:
            embeddings: Embeddings instance to use. If None, CitadelEmbeddings will be used.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use for embeddings.
            logger: Logger instance.
        """
        super().__init__(
            embeddings=embeddings,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            logger=logger
        )
        
        # Initialize storage
        self.texts = []
        self.embeddings_list = []
        self.metadatas = []
        self.ids = []
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add.
            metadatas: List of metadata dictionaries.
            ids: List of IDs for the texts.
            **kwargs: Additional parameters.
            
        Returns:
            List of IDs of the added texts.
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(i + len(self.ids)) for i in range(len(texts))]
        
        # Use empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add to storage
        self.texts.extend(texts)
        self.embeddings_list.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        return ids
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
            ids: List of IDs for the documents.
            **kwargs: Additional parameters.
            
        Returns:
            List of IDs of the added documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return self.add_texts(texts, metadatas, ids, **kwargs)
    
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
        if not self.embeddings_list:
            return []
        
        # Get documents and scores
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        
        # Return just the documents
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query and return with scores.
        
        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Additional parameters.
            
        Returns:
            List of tuples of (document, score).
        """
        if not self.embeddings_list:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Convert to numpy arrays for efficient computation
        query_embedding_np = np.array(query_embedding)
        embeddings_np = np.array(self.embeddings_list)
        
        # Compute cosine similarity
        dot_products = np.dot(embeddings_np, query_embedding_np)
        query_norm = np.linalg.norm(query_embedding_np)
        embedding_norms = np.linalg.norm(embeddings_np, axis=1)
        
        # Avoid division by zero
        cosine_similarities = dot_products / (query_norm * embedding_norms + 1e-10)
        
        # Get top k indices
        top_k_indices = np.argsort(cosine_similarities)[-k:][::-1]
        
        # Create result
        result = []
        for idx in top_k_indices:
            doc = Document(
                page_content=self.texts[idx],
                metadata=self.metadatas[idx]
            )
            score = cosine_similarities[idx]
            result.append((doc, score))
        
        return result
    
    def save(self, path: str, **kwargs) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save to.
            **kwargs: Additional parameters.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the data
        data = {
            "texts": self.texts,
            "embeddings": self.embeddings_list,
            "metadatas": self.metadatas,
            "ids": self.ids
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(
        cls,
        path: str,
        embeddings: Optional[Embeddings] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        **kwargs
    ) -> "InMemoryVectorStore":
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load from.
            embeddings: Embeddings instance to use. If None, CitadelEmbeddings will be used.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use for embeddings.
            **kwargs: Additional parameters.
            
        Returns:
            Loaded vector store.
        """
        # Create a new instance
        vector_store = cls(
            embeddings=embeddings,
            ollama_gateway=ollama_gateway,
            model_name=model_name
        )
        
        # Load the data
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Restore the data
        vector_store.texts = data["texts"]
        vector_store.embeddings_list = data["embeddings"]
        vector_store.metadatas = data["metadatas"]
        vector_store.ids = data["ids"]
        
        return vector_store
