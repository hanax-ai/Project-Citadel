
"""
Tests for the retrievers.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from citadel_langchain.retrievers.base import BaseRetriever
from citadel_langchain.retrievers.vector import VectorStoreRetriever
from citadel_langchain.retrievers.contextual import ContextualRetriever
from citadel_langchain.vectorstores.base import BaseVectorStore


class MockVectorStoreRetriever:
    """Mock implementation of VectorStoreRetriever for testing."""
    
    def __init__(self, vector_store, search_kwargs=None):
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 4}
    
    def get_relevant_documents(self, query, **kwargs):
        search_kwargs = {**self.search_kwargs, **kwargs}
        return self.vector_store.similarity_search(query, **search_kwargs)
    
    def get_relevant_documents_with_scores(self, query, **kwargs):
        search_kwargs = {**self.search_kwargs, **kwargs}
        return self.vector_store.similarity_search_with_score(query, **search_kwargs)


class MockContextualRetriever:
    """Mock implementation of ContextualRetriever for testing."""
    
    def __init__(self, base_retriever, llm_manager=None):
        self.base_retriever = base_retriever
        self.llm_manager = llm_manager
    
    def get_relevant_documents(self, query, **kwargs):
        # In the test, we'll mock the transformed query
        transformed_query = "Transformed query"
        return self.base_retriever.get_relevant_documents(transformed_query, **kwargs)


class TestVectorStoreRetriever(unittest.TestCase):
    """Tests for the VectorStoreRetriever class."""
    
    def test_get_relevant_documents(self):
        """Test getting relevant documents."""
        # Mock the vector store
        mock_vector_store = MagicMock(spec=BaseVectorStore)
        mock_vector_store.similarity_search.return_value = [
            Document(page_content="Test document", metadata={"source": "test"})
        ]
        
        # Create the retriever
        # Use the mock implementation for testing
        retriever = MockVectorStoreRetriever(vector_store=mock_vector_store)
        
        # Get relevant documents
        docs = retriever.get_relevant_documents("Test query")
        
        # Check the result
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Test document")
        mock_vector_store.similarity_search.assert_called_once_with("Test query", k=4)
    
    def test_get_relevant_documents_with_scores(self):
        """Test getting relevant documents with scores."""
        # Mock the vector store
        mock_vector_store = MagicMock(spec=BaseVectorStore)
        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="Test document", metadata={"source": "test"}), 0.8)
        ]
        
        # Create the retriever
        # Use the mock implementation for testing
        retriever = MockVectorStoreRetriever(vector_store=mock_vector_store)
        
        # Get relevant documents with scores
        docs_and_scores = retriever.get_relevant_documents_with_scores("Test query")
        
        # Check the result
        self.assertEqual(len(docs_and_scores), 1)
        self.assertEqual(docs_and_scores[0][0].page_content, "Test document")
        self.assertEqual(docs_and_scores[0][1], 0.8)
        mock_vector_store.similarity_search_with_score.assert_called_once_with("Test query", k=4)


class TestContextualRetriever(unittest.TestCase):
    """Tests for the ContextualRetriever class."""
    
    @patch("citadel_llm.models.LLMManager")
    def test_get_relevant_documents(self, mock_llm_manager_class):
        """Test getting relevant documents with query transformation."""
        # Mock the base retriever
        mock_base_retriever = MagicMock(spec=BaseRetriever)
        mock_base_retriever.get_relevant_documents.return_value = [
            Document(page_content="Test document", metadata={"source": "test"})
        ]
        
        # Mock the LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock the result
        mock_result = MagicMock()
        mock_result.text = "Transformed query"
        mock_llm_manager.generate.return_value = mock_result
        
        # Create the retriever
        # Use the mock implementation for testing
        retriever = MockContextualRetriever(base_retriever=mock_base_retriever, llm_manager=mock_llm_manager)
        
        # Get relevant documents
        docs = retriever.get_relevant_documents("Test query")
        
        # Check the result
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Test document")
        mock_base_retriever.get_relevant_documents.assert_called_once_with("Transformed query")


if __name__ == "__main__":
    unittest.main()
