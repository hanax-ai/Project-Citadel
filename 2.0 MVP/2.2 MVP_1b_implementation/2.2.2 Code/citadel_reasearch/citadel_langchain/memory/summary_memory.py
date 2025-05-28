
"""
Summary memory for Project Citadel LangChain integration.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain.memory import ConversationSummaryMemory as LangChainSummaryMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from .base_memory import BaseMemory


class SummaryMemory(BaseMemory):
    """
    Summary memory for maintaining condensed context.
    
    This memory component maintains a summary of the conversation history.
    """
    
    def __init__(
        self,
        memory: Optional[LangChainSummaryMemory] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        memory_key: str = "summary",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = True,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        prompt: Optional[str] = None,
        summarize_step: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the summary memory.
        
        Args:
            memory: LangChain summary memory to use. If None, a new one will be created.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            llm_manager: LLM manager to use. If None, a new one will be created.
            model_name: Name of the model to use.
            memory_key: Key to use for storing memory in the chain.
            input_key: Key to use for storing inputs in the memory.
            output_key: Key to use for storing outputs in the memory.
            return_messages: Whether to return messages or a string.
            human_prefix: Prefix for human messages.
            ai_prefix: Prefix for AI messages.
            prompt: Prompt to use for summarization.
            summarize_step: Number of exchanges before summarizing.
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
        self.prompt = prompt
        self.summarize_step = summarize_step
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=self.ollama_gateway
        )
        
        # Create the LangChain summary memory if not provided
        if not self._memory:
            from langchain.llms.ollama import Ollama
            
            # Create an LLM for summarization
            llm = Ollama(
                model=model_name,
                base_url=self.ollama_gateway.base_url,
            )
            
            self._memory = LangChainSummaryMemory(
                llm=llm,
                memory_key=memory_key,
                input_key=input_key,
                output_key=output_key,
                return_messages=return_messages,
                human_prefix=human_prefix,
                ai_prefix=ai_prefix,
                prompt=prompt,
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
    def summary(self) -> str:
        """
        Get the summary.
        
        Returns:
            Summary of the conversation.
        """
        return self._memory.moving_summary_buffer
    
    def predict_new_summary(
        self,
        messages: List[BaseMessage],
        existing_summary: str
    ) -> str:
        """
        Predict a new summary.
        
        Args:
            messages: List of messages.
            existing_summary: Existing summary.
            
        Returns:
            New summary.
        """
        return self._memory.predict_new_summary(messages, existing_summary)
    
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
            "prompt": self.prompt,
            "summarize_step": self.summarize_step,
        })
        return node
