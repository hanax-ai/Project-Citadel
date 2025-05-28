
"""
FAISS vector store for Project Citadel LangChain integration.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway

from .base import BaseVectorStore, CitadelEmbeddings


class FAISSVectorStore(BaseVectorStore):
    """
    Vector store that uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        index: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            embeddings: Embeddings instance to use. If None, CitadelEmbeddings will be used.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use for embeddings.
            index: FAISS index to use. If None, a new one will be created.
            logger: Logger instance.
        """
        super().__init__(
            embeddings=embeddings,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            logger=logger
        )
        
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )
        
        self.faiss = faiss
        
        # Initialize storage
        self.texts = []
        self.metadatas = []
        self.ids = []
        
        # Initialize index
        self.index = index
    
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
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Initialize index if it doesn't exist
        if self.index is None:
            dim = embeddings_np.shape[1]
            self.index = self.faiss.IndexFlatL2(dim)
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Add to storage
        self.texts.extend(texts)
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
        if not self.index or self.index.ntotal == 0:
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
        if not self.index or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Convert to numpy array
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        # Search index
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Create result
        result = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                doc = Document(
                    page_content=self.texts[idx],
                    metadata=self.metadatas[idx]
                )
                score = float(distances[0][i])
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
        
        # Save the index
        index_path = f"{path}.index"
        self.faiss.write_index(self.index, index_path)
        
        # Save the data
        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "ids": self.ids
        }
        
        data_path = f"{path}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(
        cls,
        path: str,
        embeddings: Optional[Embeddings] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        **kwargs
    ) -> "FAISSVectorStore":
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
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Could not import faiss. Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )
        
        # Load the index
        index_path = f"{path}.index"
        index = faiss.read_index(index_path)
        
        # Load the data
        data_path = f"{path}.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        # Create a new instance
        vector_store = cls(
            embeddings=embeddings,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            index=index
        )
        
        # Restore the data
        vector_store.texts = data["texts"]
        vector_store.metadatas = data["metadatas"]
        vector_store.ids = data["ids"]
        
        return vector_store
