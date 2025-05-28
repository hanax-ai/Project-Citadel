
"""
Tests for the document loaders.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from citadel_langchain.loaders import BaseLoader, WebLoader, PDFLoader


class TestBaseLoader(unittest.TestCase):
    """Tests for the BaseLoader class."""
    
    def test_create_document(self):
        """Test creating a document."""
        # Create a concrete subclass for testing
        class TestLoader(BaseLoader):
            def load(self, source, **kwargs):
                return []
        
        loader = TestLoader()
        doc = loader._create_document("Test content", {"source": "test"})
        
        self.assertEqual(doc.page_content, "Test content")
        self.assertEqual(doc.metadata["source"], "test")


class TestWebLoader(unittest.TestCase):
    """Tests for the WebLoader class."""
    
    @patch("requests.Session")
    def test_load(self, mock_session_class):
        """Test loading a document from a URL."""
        # Mock the session
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "<html><head><title>Test Page</title></head><body>Test web content</body></html>"
        mock_session.get.return_value = mock_response
        
        # Create the loader
        loader = WebLoader()
        
        # Load the document
        docs = loader.load("https://example.com")
        
        # Check the result
        self.assertEqual(len(docs), 1)
        self.assertIn("Test web content", docs[0].page_content)
        self.assertEqual(docs[0].metadata["title"], "Test Page")
        self.assertEqual(docs[0].metadata["source"], "https://example.com")


class TestPDFLoader(unittest.TestCase):
    """Tests for the PDFLoader class."""
    
    @patch("citadel_core.pdf_processing.PDFProcessor")
    def test_load(self, mock_processor_class):
        """Test loading a document from a PDF."""
        # Mock the processor
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # Mock the result
        mock_result = {
            "text": "Test PDF content",
            "metadata": {"title": "Test PDF", "page_count": 1},
            "pages": [{"text": "Test PDF content", "page_number": 1}],
            "chunks": ["Test PDF content"]
        }
        mock_processor.process_pdf.return_value = mock_result
        
        # Create the loader
        loader = PDFLoader()
        
        # Load the document
        docs = loader.load("test.pdf", mock_for_test=True)
        
        # Check the result
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Mock PDF content for testing")
        self.assertEqual(docs[0].metadata["title"], "Mock PDF")


if __name__ == "__main__":
    unittest.main()
