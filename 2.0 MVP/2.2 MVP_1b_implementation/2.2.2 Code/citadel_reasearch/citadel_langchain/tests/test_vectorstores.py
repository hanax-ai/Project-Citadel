
"""
Tests for the vector stores.
"""

import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from langchain_core.documents import Document

from citadel_langchain.vectorstores import BaseVectorStore, InMemoryVectorStore, FAISSVectorStore
from citadel_langchain.vectorstores.base import CitadelEmbeddings


class TestCitadelEmbeddings(unittest.TestCase):
    """Tests for the CitadelEmbeddings class."""
    
    @patch("citadel_llm.gateway.OllamaGateway")
    def test_embed_documents(self, mock_gateway_class):
        """Test embedding documents."""
        # Mock the gateway
        mock_gateway = MagicMock()
        mock_gateway_class.return_value = mock_gateway
        
        # Mock the result
        mock_result = {"embedding": [0.1, 0.2, 0.3]}
        mock_gateway.generate_embeddings.return_value = mock_result
        
        # Create the embeddings
        embeddings = CitadelEmbeddings()
        
        # Patch asyncio.run to return the mock result
        with patch("asyncio.new_event_loop") as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = [[0.1, 0.2, 0.3]]
            
            # Embed the documents
            result = embeddings.embed_documents(["Test document"])
            
            # Check the result
            self.assertEqual(result, [[0.1, 0.2, 0.3]])


class TestInMemoryVectorStore(unittest.TestCase):
    """Tests for the InMemoryVectorStore class."""
    
    @patch("citadel_langchain.vectorstores.base.CitadelEmbeddings")
    def test_add_texts(self, mock_embeddings_class):
        """Test adding texts to the vector store."""
        # Mock the embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # Create the vector store
        vector_store = InMemoryVectorStore()
        
        # Add texts
        ids = vector_store.add_texts(["Test document"])
        
        # Check the result
        self.assertEqual(len(ids), 1)
        self.assertEqual(vector_store.texts, ["Test document"])
        self.assertEqual(vector_store.embeddings_list, [[0.1, 0.2, 0.3]])
    
    @patch("citadel_langchain.vectorstores.base.CitadelEmbeddings")
    def test_similarity_search(self, mock_embeddings_class):
        """Test similarity search."""
        # Mock the embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Create the vector store
        vector_store = InMemoryVectorStore()
        
        # Add texts
        vector_store.add_texts(["Test document"])
        
        # Search
        results = vector_store.similarity_search("Test query")
        
        # Check the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Test document")
    
    @patch("citadel_langchain.vectorstores.base.CitadelEmbeddings")
    def test_save_and_load(self, mock_embeddings_class):
        """Test saving and loading the vector store."""
        # Mock the embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile() as temp_file:
            # Create the vector store
            vector_store = InMemoryVectorStore()
            
            # Add texts
            vector_store.add_texts(["Test document"])
            
            # Save
            vector_store.save(temp_file.name)
            
            # Load
            loaded_store = InMemoryVectorStore.load(temp_file.name)
            
            # Check the result
            self.assertEqual(loaded_store.texts, ["Test document"])
            self.assertEqual(loaded_store.embeddings_list, [[0.1, 0.2, 0.3]])


class TestFAISSVectorStore(unittest.TestCase):
    """Tests for the FAISSVectorStore class."""
    
    @patch("citadel_langchain.vectorstores.base.CitadelEmbeddings")
    @patch("faiss.IndexFlatL2")
    def test_add_texts(self, mock_index_class, mock_embeddings_class):
        """Test adding texts to the vector store."""
        # Mock the embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock the index
        mock_index = MagicMock()
        mock_index_class.return_value = mock_index
        
        # Create the vector store
        vector_store = FAISSVectorStore()
        
        # Add texts
        ids = vector_store.add_texts(["Test document"])
        
        # Check the result
        self.assertEqual(len(ids), 1)
        self.assertEqual(vector_store.texts, ["Test document"])
        mock_index.add.assert_called_once()
    
    @patch("citadel_langchain.vectorstores.base.CitadelEmbeddings")
    @patch("faiss.IndexFlatL2")
    def test_similarity_search(self, mock_index_class, mock_embeddings_class):
        """Test similarity search."""
        # Mock the embeddings
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock the index
        mock_index = MagicMock()
        mock_index_class.return_value = mock_index
        mock_index.ntotal = 1
        mock_index.search.return_value = (np.array([[0.5]]), np.array([[0]]))
        
        # Create the vector store
        vector_store = FAISSVectorStore()
        vector_store.index = mock_index
        
        # Add texts
        vector_store.add_texts(["Test document"])
        
        # Search
        results = vector_store.similarity_search("Test query")
        
        # Check the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Test document")


if __name__ == "__main__":
    unittest.main()
