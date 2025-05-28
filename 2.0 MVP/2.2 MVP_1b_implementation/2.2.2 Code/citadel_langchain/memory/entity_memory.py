
"""
Entity memory for Project Citadel LangChain integration.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

from langchain.memory import ConversationEntityMemory as LangChainEntityMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from .base_memory import BaseMemory


class EntityMemory(BaseMemory):
    """
    Entity memory for tracking entities mentioned in conversations.
    
    This memory component keeps track of entities mentioned in the conversation.
    """
    
    def __init__(
        self,
        memory: Optional[LangChainEntityMemory] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        memory_key: str = "entities",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = True,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        k: int = 3,
        chat_history_key: str = "history",
        entity_cache: Optional[Dict[str, Any]] = None,
        entity_extraction_prompt: Optional[str] = None,
        entity_summarization_prompt: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the entity memory.
        
        Args:
            memory: LangChain entity memory to use. If None, a new one will be created.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            llm_manager: LLM manager to use. If None, a new one will be created.
            model_name: Name of the model to use.
            memory_key: Key to use for storing memory in the chain.
            input_key: Key to use for storing inputs in the memory.
            output_key: Key to use for storing outputs in the memory.
            return_messages: Whether to return messages or a string.
            human_prefix: Prefix for human messages.
            ai_prefix: Prefix for AI messages.
            k: Number of previous exchanges to keep in memory.
            chat_history_key: Key to use for storing chat history.
            entity_cache: Cache of entities.
            entity_extraction_prompt: Prompt to use for entity extraction.
            entity_summarization_prompt: Prompt to use for entity summarization.
            logger: Logger instance.
        """
        super().__init__(
            memory=memory,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            memory_key=memory_key,
            input_key=input_key,
            output_key=output_key,
            return_messages=return_messages,
            logger=logger,
        )
        
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.k = k
        self.chat_history_key = chat_history_key
        self.entity_cache = entity_cache or {}
        self.entity_extraction_prompt = entity_extraction_prompt
        self.entity_summarization_prompt = entity_summarization_prompt
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=self.ollama_gateway
        )
        
        # Create the LangChain entity memory if not provided
        if not self._memory:
            from langchain.llms.ollama import Ollama
            
            # Create an LLM for entity extraction and summarization
            llm = Ollama(
                model=model_name,
                base_url=self.ollama_gateway.base_url,
            )
            
            self._memory = LangChainEntityMemory(
                llm=llm,
                memory_key=memory_key,
                input_key=input_key,
                output_key=output_key,
                return_messages=return_messages,
                human_prefix=human_prefix,
                ai_prefix=ai_prefix,
                k=k,
                chat_history_key=chat_history_key,
                entity_cache=entity_cache,
                entity_extraction_prompt=entity_extraction_prompt,
                entity_summarization_prompt=entity_summarization_prompt,
            )
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables.
        
        Args:
            inputs: Input values.
            
        Returns:
            Dictionary of memory variables.
        """
        return self._memory.load_memory_variables(inputs)
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save the context.
        
        Args:
            inputs: Input values.
            outputs: Output values.
        """
        self._memory.save_context(inputs, outputs)
    
    def clear(self) -> None:
        """
        Clear the memory.
        """
        self._memory.clear()
    
    @property
    def buffer(self) -> Union[List[BaseMessage], str]:
        """
        Get the buffer.
        
        Returns:
            Buffer as a list of messages or a string.
        """
        return self._memory.buffer
    
    @property
    def entity_store(self) -> Dict[str, Any]:
        """
        Get the entity store.
        
        Returns:
            Entity store.
        """
        return self._memory.entity_store
    
    def get_entity_summaries(self, entities: List[str]) -> Dict[str, str]:
        """
        Get entity summaries.
        
        Args:
            entities: List of entities.
            
        Returns:
            Dictionary mapping entities to their summaries.
        """
        return {
            entity: self.entity_store.get(entity, "")
            for entity in entities
        }
    
    def to_graph_node(self) -> Dict[str, Any]:
        """
        Convert the memory to a graph node.
        
        Returns:
            Dictionary representation of the memory as a graph node.
        """
        node = super().to_graph_node()
        node.update({
            "human_prefix": self.human_prefix,
            "ai_prefix": self.ai_prefix,
            "k": self.k,
            "chat_history_key": self.chat_history_key,
        })
        return node
