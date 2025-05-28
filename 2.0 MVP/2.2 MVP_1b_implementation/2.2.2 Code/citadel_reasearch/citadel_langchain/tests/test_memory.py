
"""
Tests for memory components.
"""

import os
import tempfile
import unittest
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

from citadel_langchain.memory import (
    BaseMemory,
    BufferMemory,
    ConversationMemory,
    SummaryMemory,
    EntityMemory,
)


class TestBufferMemory(unittest.TestCase):
    """Test cases for BufferMemory."""
    
    def setUp(self):
        """Set up test cases."""
        self.memory = BufferMemory()
    
    def test_save_context(self):
        """Test saving context."""
        self.memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )
        
        variables = self.memory.load_memory_variables({})
        self.assertIn("history", variables)
        
        if self.memory.return_messages:
            messages = variables["history"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0].content, "Hello")
            self.assertEqual(messages[1].content, "Hi there!")
        else:
            history = variables["history"]
            self.assertIn("Hello", history)
            self.assertIn("Hi there!", history)
    
    def test_clear(self):
        """Test clearing memory."""
        self.memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )
        
        self.memory.clear()
        
        variables = self.memory.load_memory_variables({})
        if self.memory.return_messages:
            self.assertEqual(len(variables["history"]), 0)
        else:
            self.assertEqual(variables["history"], "")
    
    def test_save_load(self):
        """Test saving and loading memory."""
        self.memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.memory.save(f.name)
            
            loaded_memory = BufferMemory.load(f.name)
            
            variables = loaded_memory.load_memory_variables({})
            self.assertIn("history", variables)
            
            if loaded_memory.return_messages:
                messages = variables["history"]
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0].content, "Hello")
                self.assertEqual(messages[1].content, "Hi there!")
            else:
                history = variables["history"]
                self.assertIn("Hello", history)
                self.assertIn("Hi there!", history)
            
            os.unlink(f.name)


class TestConversationMemory(unittest.TestCase):
    """Test cases for ConversationMemory."""
    
    def setUp(self):
        """Set up test cases."""
        self.memory = ConversationMemory(k=2)
    
    def test_save_context(self):
        """Test saving context."""
        # Add first exchange
        self.memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )
        
        # Add second exchange
        self.memory.save_context(
            {"input": "How are you?"},
            {"output": "I'm doing well, thanks!"}
        )
        
        # Add third exchange (should only keep the last 2)
        self.memory.save_context(
            {"input": "What's your name?"},
            {"output": "I'm an AI assistant."}
        )
        
        variables = self.memory.load_memory_variables({})
        self.assertIn("conversation", variables)
        
        if self.memory.return_messages:
            messages = variables["conversation"]
            self.assertEqual(len(messages), 4)  # 2 exchanges = 4 messages
            self.assertEqual(messages[0].content, "How are you?")
            self.assertEqual(messages[1].content, "I'm doing well, thanks!")
            self.assertEqual(messages[2].content, "What's your name?")
            self.assertEqual(messages[3].content, "I'm an AI assistant.")
        else:
            conversation = variables["conversation"]
            self.assertIn("How are you?", conversation)
            self.assertIn("I'm doing well, thanks!", conversation)
            self.assertIn("What's your name?", conversation)
            self.assertIn("I'm an AI assistant.", conversation)
            self.assertNotIn("Hello", conversation)
            self.assertNotIn("Hi there!", conversation)
    
    def test_clear(self):
        """Test clearing memory."""
        self.memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"}
        )
        
        self.memory.clear()
        
        variables = self.memory.load_memory_variables({})
        if self.memory.return_messages:
            self.assertEqual(len(variables["conversation"]), 0)
        else:
            self.assertEqual(variables["conversation"], "")


class TestSummaryMemory(unittest.TestCase):
    """Test cases for SummaryMemory."""
    
    def setUp(self):
        """Set up test cases."""
        # Skip tests if no Ollama server is available
        try:
            from langchain.llms.ollama import Ollama
            llm = Ollama(model="mistral:latest")
            llm.invoke("Hello")
        except:
            self.skipTest("Ollama server not available")
        
        self.memory = SummaryMemory()
    
    def test_save_context(self):
        """Test saving context."""
        self.memory.save_context(
            {"input": "My name is John and I live in New York."},
            {"output": "Nice to meet you, John! New York is a great city."}
        )
        
        variables = self.memory.load_memory_variables({})
        self.assertIn("summary", variables)
        
        # The summary should contain information about John and New York
        summary = variables["summary"]
        self.assertIsInstance(summary, str)
        
        # Add more context
        self.memory.save_context(
            {"input": "I work as a software engineer."},
            {"output": "That's interesting! What kind of software do you work on?"}
        )
        
        # The summary should be updated
        variables = self.memory.load_memory_variables({})
        self.assertIn("summary", variables)
    
    def test_clear(self):
        """Test clearing memory."""
        self.memory.save_context(
            {"input": "My name is John and I live in New York."},
            {"output": "Nice to meet you, John! New York is a great city."}
        )
        
        self.memory.clear()
        
        variables = self.memory.load_memory_variables({})
        self.assertEqual(variables["summary"], "")


class TestEntityMemory(unittest.TestCase):
    """Test cases for EntityMemory."""
    
    def setUp(self):
        """Set up test cases."""
        # Skip tests if no Ollama server is available
        try:
            from langchain.llms.ollama import Ollama
            llm = Ollama(model="mistral:latest")
            llm.invoke("Hello")
        except:
            self.skipTest("Ollama server not available")
        
        self.memory = EntityMemory()
    
    def test_save_context(self):
        """Test saving context."""
        self.memory.save_context(
            {"input": "My name is John and I live in New York."},
            {"output": "Nice to meet you, John! New York is a great city."}
        )
        
        variables = self.memory.load_memory_variables({
            "input": "Tell me about John."
        })
        
        self.assertIn("entities", variables)
        entities = variables["entities"]
        
        # The entity store should contain information about John
        self.assertIsInstance(entities, str)
        
        # Add more context
        self.memory.save_context(
            {"input": "I also have a friend named Sarah who lives in Boston."},
            {"output": "That's nice! How long have you known Sarah?"}
        )
        
        # The entity store should be updated with information about Sarah
        variables = self.memory.load_memory_variables({
            "input": "Tell me about Sarah."
        })
        
        self.assertIn("entities", variables)
    
    def test_clear(self):
        """Test clearing memory."""
        self.memory.save_context(
            {"input": "My name is John and I live in New York."},
            {"output": "Nice to meet you, John! New York is a great city."}
        )
        
        self.memory.clear()
        
        variables = self.memory.load_memory_variables({
            "input": "Tell me about John."
        })
        
        self.assertEqual(variables["entities"], "")


if __name__ == "__main__":
    unittest.main()
