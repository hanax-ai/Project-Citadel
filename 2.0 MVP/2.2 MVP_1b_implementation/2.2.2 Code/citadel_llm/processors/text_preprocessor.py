
"""
Text preprocessor for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Union, Callable

from .base import BaseProcessor
from .text_cleaner import TextCleaner
from .text_normalizer import TextNormalizer


class TextPreprocessor(BaseProcessor):
    """
    A processor that combines multiple text processing steps
    into a single preprocessing pipeline.
    """
    
    def __init__(
        self,
        processors: Optional[List[BaseProcessor]] = None,
        custom_processors: Optional[List[Callable[[str], str]]] = None,
        **kwargs
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            processors: List of processor instances to apply in sequence.
            custom_processors: List of custom processing functions.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        # Initialize default processors if none provided
        if processors is None:
            self.processors = [
                TextCleaner(remove_html=True, remove_extra_whitespace=True),
                TextNormalizer(normalize_whitespace=True, normalize_unicode=True)
            ]
        else:
            self.processors = processors
        
        self.custom_processors = custom_processors or []
    
    async def process(self, text: str, **kwargs) -> str:
        """
        Preprocess the input text by applying all processors in sequence.
        
        Args:
            text: Input text to preprocess.
            **kwargs: Additional processing parameters.
            
        Returns:
            Preprocessed text.
        """
        if not text:
            return ""
        
        # Apply all processors in sequence
        processed_text = text
        
        for processor in self.processors:
            processed_text = await processor.process(processed_text, **kwargs)
        
        # Apply custom processors
        for custom_processor in self.custom_processors:
            processed_text = custom_processor(processed_text)
        
        return processed_text
    
    def add_processor(self, processor: BaseProcessor) -> None:
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Processor instance to add.
        """
        self.processors.append(processor)
    
    def add_custom_processor(self, processor: Callable[[str], str]) -> None:
        """
        Add a custom processor function to the pipeline.
        
        Args:
            processor: Custom processor function to add.
        """
        self.custom_processors.append(processor)
