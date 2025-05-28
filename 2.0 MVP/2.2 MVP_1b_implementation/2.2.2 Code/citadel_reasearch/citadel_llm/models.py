
"""
Model definitions and configurations for the Citadel LLM package.

This module provides model definitions, configurations, and an LLMManager class
for managing models and processing text.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, AsyncIterator

from citadel_core.logging import get_logger
from citadel_core.config import Config
from citadel_core.utils import retry

from .exceptions import ModelNotFoundError, InvalidRequestError
from .gateway import OllamaGateway


class ModelProvider(str, Enum):
    """Enum for model providers."""
    OLLAMA = "ollama"
    # Add more providers as needed


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    
    name: str
    provider: ModelProvider
    context_window: int
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_max_tokens: Optional[int] = None
    supports_json_mode: bool = False
    supports_vision: bool = False
    supports_function_calling: bool = False
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model config to a dictionary."""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "context_window": self.context_window,
            "default_temperature": self.default_temperature,
            "default_top_p": self.default_top_p,
            "default_top_k": self.default_top_k,
            "default_max_tokens": self.default_max_tokens,
            "supports_json_mode": self.supports_json_mode,
            "supports_vision": self.supports_vision,
            "supports_function_calling": self.supports_function_calling,
            "description": self.description,
            "tags": self.tags,
        }


# Define supported models
SUPPORTED_MODELS = {
    "deepcoder:14b": ModelConfig(
        name="deepcoder:14b",
        provider=ModelProvider.OLLAMA,
        context_window=8192,
        default_temperature=0.5,
        supports_json_mode=True,
        description="DeepCoder 14B model for code generation",
        tags=["code", "programming"]
    ),
    "deepcoder:latest": ModelConfig(
        name="deepcoder:latest",
        provider=ModelProvider.OLLAMA,
        context_window=8192,
        default_temperature=0.5,
        supports_json_mode=True,
        description="Latest DeepCoder model for code generation",
        tags=["code", "programming"]
    ),
    "deepseek-r1:32b": ModelConfig(
        name="deepseek-r1:32b",
        provider=ModelProvider.OLLAMA,
        context_window=16384,
        default_temperature=0.7,
        supports_json_mode=True,
        description="DeepSeek R1 32B model for general purpose tasks",
        tags=["general", "reasoning"]
    ),
    "deepseek-r1:latest": ModelConfig(
        name="deepseek-r1:latest",
        provider=ModelProvider.OLLAMA,
        context_window=16384,
        default_temperature=0.7,
        supports_json_mode=True,
        description="Latest DeepSeek R1 model for general purpose tasks",
        tags=["general", "reasoning"]
    ),
    "mistral:latest": ModelConfig(
        name="mistral:latest",
        provider=ModelProvider.OLLAMA,
        context_window=8192,
        default_temperature=0.7,
        supports_json_mode=True,
        description="Latest Mistral model for general purpose tasks",
        tags=["general", "reasoning"]
    ),
    "deepcoder-bf16:latest": ModelConfig(
        name="deepcoder-bf16:latest",
        provider=ModelProvider.OLLAMA,
        context_window=8192,
        default_temperature=0.5,
        supports_json_mode=True,
        description="Latest DeepCoder model with BF16 precision for code generation",
        tags=["code", "programming"]
    ),
}


@dataclass
class GenerationOptions:
    """Options for text generation."""
    
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_mode: bool = False
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the generation options to a dictionary."""
        result = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        
        if self.max_tokens is not None:
            result["num_predict"] = self.max_tokens
        
        if self.stop_sequences:
            result["stop"] = self.stop_sequences
        
        if self.frequency_penalty != 0.0:
            result["frequency_penalty"] = self.frequency_penalty
        
        if self.presence_penalty != 0.0:
            result["presence_penalty"] = self.presence_penalty
        
        if self.seed is not None:
            result["seed"] = self.seed
        
        return result


@dataclass
class Message:
    """A message in a conversation."""
    
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class GenerationResult:
    """Result of a text generation."""
    
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Optional[str] = None
    
    @classmethod
    def from_ollama_response(cls, response: Dict[str, Any], text: str) -> 'GenerationResult':
        """Create a GenerationResult from an Ollama response."""
        return cls(
            text=text,
            model=response.get("model", ""),
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            finish_reason=response.get("done", False) and "stop" or None
        )


class LLMManager:
    """Manager for language models."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        gateway: Optional[OllamaGateway] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM manager.
        
        Args:
            config: Configuration object.
            gateway: Ollama gateway client.
            logger: Logger instance.
        """
        self.config = config or Config()
        self.logger = logger or get_logger("citadel.llm")
        
        # Initialize gateway
        gateway_config = self.config.get("llm", {}).get("gateway", {})
        self.gateway = gateway or OllamaGateway(
            base_url=gateway_config.get("base_url", "http://localhost:11434"),
            timeout=gateway_config.get("timeout", 30),
            logger=self.logger
        )
        
        # Initialize model configs
        self.models = SUPPORTED_MODELS.copy()
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get the configuration for a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Model configuration.
            
        Raises:
            ModelNotFoundError: If the model is not found.
        """
        if model_name not in self.models:
            raise ModelNotFoundError(model_name)
        
        return self.models[model_name]
    
    def list_models(self) -> List[ModelConfig]:
        """
        List all available models.
        
        Returns:
            List of model configurations.
        """
        return list(self.models.values())
    
    async def refresh_models(self) -> List[str]:
        """
        Refresh the list of available models from the gateway.
        
        Returns:
            List of model names.
        """
        models = await self.gateway.list_models()
        return [model["name"] for model in models]
    
    async def generate(
        self,
        prompt: str,
        model_name: str,
        options: Optional[GenerationOptions] = None,
        system_message: Optional[str] = None,
        stream: bool = False
    ) -> Union[GenerationResult, AsyncIterator[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt.
            model_name: Name of the model to use.
            options: Generation options.
            system_message: System message to prepend to the prompt.
            stream: Whether to stream the response.
            
        Returns:
            If stream is False, returns a GenerationResult.
            If stream is True, returns an AsyncIterator of strings.
            
        Raises:
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        # Validate model
        if model_name not in self.models:
            raise ModelNotFoundError(model_name)
        
        model_config = self.models[model_name]
        
        # Use default options if not provided
        if options is None:
            options = GenerationOptions(
                temperature=model_config.default_temperature,
                top_p=model_config.default_top_p,
                top_k=model_config.default_top_k,
                max_tokens=model_config.default_max_tokens
            )
        
        # Generate text
        if stream:
            return await self.gateway.generate_stream(
                prompt=prompt,
                model=model_name,
                options=options.to_dict(),
                system=system_message
            )
        else:
            response = await self.gateway.generate(
                prompt=prompt,
                model=model_name,
                options=options.to_dict(),
                system=system_message
            )
            
            return GenerationResult.from_ollama_response(response, response.get("response", ""))
    
    async def chat(
        self,
        messages: List[Message],
        model_name: str,
        options: Optional[GenerationOptions] = None,
        stream: bool = False
    ) -> Union[GenerationResult, AsyncIterator[str]]:
        """
        Generate a chat response.
        
        Args:
            messages: List of messages in the conversation.
            model_name: Name of the model to use.
            options: Generation options.
            stream: Whether to stream the response.
            
        Returns:
            If stream is False, returns a GenerationResult.
            If stream is True, returns an AsyncIterator of strings.
            
        Raises:
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        # Validate model
        if model_name not in self.models:
            raise ModelNotFoundError(model_name)
        
        model_config = self.models[model_name]
        
        # Use default options if not provided
        if options is None:
            options = GenerationOptions(
                temperature=model_config.default_temperature,
                top_p=model_config.default_top_p,
                top_k=model_config.default_top_k,
                max_tokens=model_config.default_max_tokens
            )
        
        # Convert messages to dict
        messages_dict = [message.to_dict() for message in messages]
        
        # Generate chat response
        if stream:
            return await self.gateway.chat_stream(
                messages=messages_dict,
                model=model_name,
                options=options.to_dict()
            )
        else:
            response = await self.gateway.chat(
                messages=messages_dict,
                model=model_name,
                options=options.to_dict()
            )
            
            message_content = response.get("message", {}).get("content", "")
            return GenerationResult.from_ollama_response(response, message_content)
