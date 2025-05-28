
"""
Citadel LLM package.

This package provides LLM integration functionality for the Citadel project.
"""

__version__ = "0.2.0"

from .exceptions import (
    CitadelLLMError,
    ModelNotFoundError,
    ModelLoadError,
    GatewayConnectionError,
    GatewayTimeoutError,
    GatewayResponseError,
    InvalidRequestError,
    PromptTemplateError
)

from .gateway import OllamaGateway
from .models import (
    ModelProvider,
    ModelConfig,
    GenerationOptions,
    Message,
    GenerationResult,
    LLMManager,
    SUPPORTED_MODELS
)
from .prompts import (
    PromptTemplate,
    SYSTEM_MESSAGES,
    PROMPT_TEMPLATES,
    get_prompt_template,
    get_system_message,
    format_prompt
)

# Import content processing components
from .processors import (
    BaseProcessor,
    BaseLLMProcessor,
    TextCleaner,
    TextNormalizer,
    TextPreprocessor,
    TextChunker
)

from .extractors import (
    BaseExtractor,
    BaseLLMExtractor,
    EntityExtractor,
    KeywordExtractor,
    RelationshipExtractor,
    MetadataExtractor
)

from .summarizers import (
    BaseSummarizer,
    BaseLLMSummarizer,
    AbstractiveSummarizer,
    ExtractiveSummarizer,
    MultiLevelSummarizer
)

from .classifiers import (
    BaseClassifier,
    BaseLLMClassifier,
    TopicClassifier,
    SentimentClassifier,
    IntentClassifier,
    ContentTypeClassifier
)

__all__ = [
    # Exceptions
    "CitadelLLMError",
    "ModelNotFoundError",
    "ModelLoadError",
    "GatewayConnectionError",
    "GatewayTimeoutError",
    "GatewayResponseError",
    "InvalidRequestError",
    "PromptTemplateError",
    
    # Gateway
    "OllamaGateway",
    
    # Models
    "ModelProvider",
    "ModelConfig",
    "GenerationOptions",
    "Message",
    "GenerationResult",
    "LLMManager",
    "SUPPORTED_MODELS",
    
    # Prompts
    "PromptTemplate",
    "SYSTEM_MESSAGES",
    "PROMPT_TEMPLATES",
    "get_prompt_template",
    "get_system_message",
    "format_prompt",
    
    # Processors
    "BaseProcessor",
    "BaseLLMProcessor",
    "TextCleaner",
    "TextNormalizer",
    "TextPreprocessor",
    "TextChunker",
    
    # Extractors
    "BaseExtractor",
    "BaseLLMExtractor",
    "EntityExtractor",
    "KeywordExtractor",
    "RelationshipExtractor",
    "MetadataExtractor",
    
    # Summarizers
    "BaseSummarizer",
    "BaseLLMSummarizer",
    "AbstractiveSummarizer",
    "ExtractiveSummarizer",
    "MultiLevelSummarizer",
    
    # Classifiers
    "BaseClassifier",
    "BaseLLMClassifier",
    "TopicClassifier",
    "SentimentClassifier",
    "IntentClassifier",
    "ContentTypeClassifier"
]
