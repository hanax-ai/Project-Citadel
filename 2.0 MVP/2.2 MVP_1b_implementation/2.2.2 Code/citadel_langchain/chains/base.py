
"""
Base chain for Project Citadel LangChain integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager


class BaseChain(ABC):
    """Base class for all chains."""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        prompt: Optional[BasePromptTemplate] = None,
        output_parser: Optional[BaseOutputParser] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the chain.
        
        Args:
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            prompt: Prompt template to use.
            output_parser: Output parser to use.
            logger: Logger instance.
        """
        self.logger = logger or get_logger(f"citadel.langchain.chains.{self.__class__.__name__.lower()}")
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.prompt = prompt
        self.output_parser = output_parser
    
    @abstractmethod
    async def _arun(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the chain asynchronously.
        
        Args:
            inputs: Input values.
            **kwargs: Additional parameters.
            
        Returns:
            Output values.
        """
        pass
    
    def run(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the chain.
        
        Args:
            inputs: Input values.
            **kwargs: Additional parameters.
            
        Returns:
            Output values.
        """
        import asyncio
        
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._arun(inputs, **kwargs))
        finally:
            loop.close()
    
    async def _agenerate_from_prompt(
        self,
        prompt_values: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt_values: Values to format the prompt with.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        if not self.prompt:
            raise ValueError("Prompt template not set")
        
        # Format the prompt
        prompt_str = self.prompt.format(**prompt_values)
        
        # Generate text
        result = await self.llm_manager.generate(
            prompt=prompt_str,
            model_name=kwargs.get("model_name", self.model_name),
            stream=kwargs.get("stream", False)
        )
        
        if kwargs.get("stream", False):
            # Collect all chunks
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            return "".join(chunks)
        else:
            return result.text
    
    def _parse_output(self, output: str) -> Any:
        """
        Parse the output using the output parser.
        
        Args:
            output: Output text to parse.
            
        Returns:
            Parsed output.
        """
        if self.output_parser:
            return self.output_parser.parse(output)
        else:
            return output
