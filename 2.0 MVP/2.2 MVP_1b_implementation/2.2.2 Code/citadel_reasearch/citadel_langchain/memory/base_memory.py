
"""
Base memory class for Project Citadel LangChain integration.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.memory import BaseMemory as LangChainBaseMemory
from langchain_core.messages import BaseMessage

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway


class BaseMemory(ABC):
    """
    Abstract base class for all memory components in Project Citadel.
    
    This class provides a common interface for all memory components and
    handles integration with LangChain's memory components.
    """
    
    def __init__(
        self,
        memory: Optional[LangChainBaseMemory] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        memory_key: str = "memory",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the memory component.
        
        Args:
            memory: LangChain memory component to use. If None, a new one will be created.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            memory_key: Key to use for storing memory in the chain.
            input_key: Key to use for storing inputs in the memory.
            output_key: Key to use for storing outputs in the memory.
            return_messages: Whether to return messages or a string.
            logger: Logger instance.
        """
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.model_name = model_name
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.logger = logger or get_logger(f"citadel.langchain.memory.{self.__class__.__name__.lower()}")
        
        # The underlying LangChain memory component
        self._memory = memory
    
    @property
    def memory_variables(self) -> List[str]:
        """
        Get the memory variables.
        
        Returns:
            List of memory variables.
        """
        if self._memory:
            return self._memory.memory_variables
        return [self.memory_key]
    
    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables.
        
        Args:
            inputs: Input values.
            
        Returns:
            Dictionary of memory variables.
        """
        pass
    
    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save the context.
        
        Args:
            inputs: Input values.
            outputs: Output values.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear the memory.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the memory to disk.
        
        Args:
            path: Path to save to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the memory
        with open(path, "wb") as f:
            pickle.dump(self._memory, f)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseMemory":
        """
        Load the memory from disk.
        
        Args:
            path: Path to load from.
            **kwargs: Additional parameters.
            
        Returns:
            Loaded memory.
        """
        # Load the memory
        with open(path, "rb") as f:
            memory = pickle.load(f)
        
        # Create a new instance
        return cls(memory=memory, **kwargs)
    
    def to_graph_node(self) -> Dict[str, Any]:
        """
        Convert the memory to a graph node.
        
        Returns:
            Dictionary representation of the memory as a graph node.
        """
        return {
            "type": "memory",
            "class": self.__class__.__name__,
            "memory_key": self.memory_key,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "return_messages": self.return_messages,
        }
