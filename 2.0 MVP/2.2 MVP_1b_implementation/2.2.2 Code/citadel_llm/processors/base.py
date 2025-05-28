
"""
Base classes for text processors in the Citadel LLM package.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class BaseProcessor(ABC):
    """Base class for all text processors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the processor.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger or get_logger(f"citadel.llm.processors.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def process(self, text: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Process the input text.
        
        Args:
            text: Input text to process.
            **kwargs: Additional processing parameters.
            
        Returns:
            Processed text or a dictionary with processing results.
        """
        pass


class BaseLLMProcessor(BaseProcessor):
    """Base class for text processors that use LLMs."""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM processor.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            logger: Logger instance.
        """
        super().__init__(logger)
        
        self.llm_manager = llm_manager or LLMManager()
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.generation_options = generation_options
    
    async def _generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt.
            system_message: System message.
            stream: Whether to stream the response.
            
        Returns:
            Generated text.
        """
        try:
            result = await self.llm_manager.generate(
                prompt=prompt,
                model_name=self.model_name,
                options=self.generation_options,
                system_message=system_message,
                stream=stream
            )
            
            if stream:
                # Collect all chunks
                chunks = []
                async for chunk in result:
                    chunks.append(chunk)
                return "".join(chunks)
            else:
                return result.text
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise
