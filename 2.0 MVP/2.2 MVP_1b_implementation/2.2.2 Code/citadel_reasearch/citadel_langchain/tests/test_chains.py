
"""
Tests for the chains.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from citadel_langchain.chains import BaseChain, QAChain, SummarizationChain
from citadel_langchain.retrievers import BaseRetriever


class TestQAChain(unittest.TestCase):
    """Tests for the QAChain class."""
    
    @patch("citadel_llm.models.LLMManager")
    def test_run(self, mock_llm_manager_class):
        """Test running the QA chain."""
        # Mock the retriever
        mock_retriever = MagicMock(spec=BaseRetriever)
        mock_retriever.aget_relevant_documents.return_value = [
            Document(page_content="Test document", metadata={"source": "test"})
        ]
        
        # Mock the LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock the result
        mock_result = MagicMock()
        mock_result.text = "Test answer"
        mock_llm_manager.generate.return_value = mock_result
        
        # Create the chain
        chain = QAChain(retriever=mock_retriever)
        
        # Patch asyncio.run to return the mock result
        with patch("asyncio.new_event_loop") as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = {
                "question": "Test question",
                "answer": "Test answer",
                "context": "Document 1 (Source: test):\nTest document",
                "source_documents": [Document(page_content="Test document", metadata={"source": "test"})]
            }
            
            # Run the chain
            result = chain.run({"question": "Test question"})
            
            # Check the result
            self.assertEqual(result["answer"], "Test answer")
            self.assertEqual(result["question"], "Test question")
            self.assertTrue("context" in result)
            self.assertTrue("source_documents" in result)


class TestSummarizationChain(unittest.TestCase):
    """Tests for the SummarizationChain class."""
    
    @patch("citadel_llm.models.LLMManager")
    def test_run(self, mock_llm_manager_class):
        """Test running the summarization chain."""
        # Mock the LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock the result
        mock_result = MagicMock()
        mock_result.text = "Test summary"
        mock_llm_manager.generate.return_value = mock_result
        
        # Create the chain
        chain = SummarizationChain()
        
        # Patch asyncio.run to return the mock result
        with patch("asyncio.new_event_loop") as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = {
                "document": "Test document",
                "instructions": "Summarize",
                "summary": "Test summary"
            }
            
            # Run the chain
            result = chain.run({
                "document": "Test document",
                "instructions": "Summarize"
            })
            
            # Check the result
            self.assertEqual(result["summary"], "Test summary")
            self.assertEqual(result["document"], "Test document")
            self.assertEqual(result["instructions"], "Summarize")


if __name__ == "__main__":
    unittest.main()
