
"""
LangChain integration package for Project Citadel.

This package provides LangChain components for document processing and LLM interactions
that integrate with the existing Citadel infrastructure.
"""

__version__ = "0.1.0"

# Import memory components
from citadel_langchain.memory import (
    BaseMemory,
    BufferMemory,
    ConversationMemory,
    SummaryMemory,
    EntityMemory,
)
