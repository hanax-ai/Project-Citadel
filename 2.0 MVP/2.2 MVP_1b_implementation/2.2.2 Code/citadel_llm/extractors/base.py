
"""
Base classes for information extractors in the Citadel LLM package.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class BaseExtractor(ABC):
    """Base class for all information extractors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the extractor.
        
        Args:
            logger: Logger instance.
        """
        self.logger = logger or get_logger(f"citadel.llm.extractors.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extract information from the input text.
        
        Args:
            text: Input text to extract information from.
            **kwargs: Additional extraction parameters.
            
        Returns:
            Dictionary with extracted information.
        """
        pass


class BaseLLMExtractor(BaseExtractor):
    """Base class for information extractors that use LLMs."""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        output_format: str = "json",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM extractor.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            output_format: Output format ("json" or "text").
            logger: Logger instance.
        """
        super().__init__(logger)
        
        self.llm_manager = llm_manager or LLMManager()
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.generation_options = generation_options
        self.output_format = output_format
    
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
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a JSON response from the LLM.
        
        Args:
            response: Response text from the LLM.
            
        Returns:
            Parsed JSON as a dictionary.
        """
        try:
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in response")
                return {"raw_response": response}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {str(e)}")
            return {"raw_response": response, "error": str(e)}
