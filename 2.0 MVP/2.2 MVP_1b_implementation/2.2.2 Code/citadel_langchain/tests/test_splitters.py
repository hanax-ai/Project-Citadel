
"""
Tests for the text splitters.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from citadel_langchain.splitters import BaseSplitter, CharacterSplitter, RecursiveSplitter, TokenSplitter


class TestBaseSplitter(unittest.TestCase):
    """Tests for the BaseSplitter class."""
    
    def test_split_documents(self):
        """Test splitting documents."""
        # Create a concrete subclass for testing
        class TestSplitter(BaseSplitter):
            def split_text(self, text, **kwargs):
                return [text[:5], text[5:]]
        
        splitter = TestSplitter()
        docs = [
            Document(page_content="Hello world", metadata={"source": "test"})
        ]
        
        result = splitter.split_documents(docs)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "Hello")
        self.assertEqual(result[1].page_content, " world")
        self.assertEqual(result[0].metadata["source"], "test")
        self.assertEqual(result[0].metadata["chunk"], 1)
        self.assertEqual(result[0].metadata["total_chunks"], 2)


class TestCharacterSplitter(unittest.TestCase):
    """Tests for the CharacterSplitter class."""
    
    def test_split_text(self):
        """Test splitting text by characters."""
        splitter = CharacterSplitter(chunk_size=10, chunk_overlap=2)
        
        text = "Hello world, how are you doing today?"
        chunks = splitter.split_text(text)
        
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0], "Hello worl")
        
        # Check that chunks overlap
        self.assertTrue(chunks[0][-2:] == chunks[1][:2] or
                       chunks[0][-1:] == chunks[1][:1])


class TestRecursiveSplitter(unittest.TestCase):
    """Tests for the RecursiveSplitter class."""
    
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_split_text(self, mock_splitter_class):
        """Test splitting text recursively."""
        # Mock the splitter
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        
        splitter = RecursiveSplitter(chunk_size=10, chunk_overlap=2)
        splitter.splitter = mock_splitter  # Replace with mock
        
        text = "Hello world, how are you doing today?"
        chunks = splitter.split_text(text)
        
        self.assertEqual(chunks, ["Chunk 1", "Chunk 2"])
        mock_splitter.split_text.assert_called_once_with(text)


class TestTokenSplitter(unittest.TestCase):
    """Tests for the TokenSplitter class."""
    
    @patch("langchain_text_splitters.TokenTextSplitter")
    def test_split_text(self, mock_splitter_class):
        """Test splitting text by tokens."""
        # Mock the splitter
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        
        splitter = TokenSplitter(chunk_size=10, chunk_overlap=2)
        splitter.splitter = mock_splitter  # Replace with mock
        
        text = "Hello world, how are you doing today?"
        chunks = splitter.split_text(text)
        
        self.assertEqual(chunks, ["Chunk 1", "Chunk 2"])
        mock_splitter.split_text.assert_called_once_with(text)
    
    def test_fallback_split(self):
        """Test fallback splitting when tiktoken is not available."""
        splitter = TokenSplitter(chunk_size=10, chunk_overlap=2)
        splitter.splitter = None  # Force fallback
        
        text = "Hello world, how are you doing today?"
        chunks = splitter._fallback_split(text, 10, 2)
        
        self.assertTrue(len(chunks) > 1)
        # Check that each chunk is roughly the right size
        self.assertEqual(chunks[0], "Hello world,")


if __name__ == "__main__":
    unittest.main()
