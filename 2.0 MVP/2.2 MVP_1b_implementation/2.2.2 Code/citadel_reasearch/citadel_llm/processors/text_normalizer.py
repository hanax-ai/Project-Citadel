
"""
Text normalizer for the Citadel LLM package.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Union

from .base import BaseProcessor, BaseLLMProcessor
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class TextNormalizer(BaseProcessor):
    """
    A processor for normalizing text by standardizing characters,
    case, whitespace, etc.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_accents: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
        normalize_quotes: bool = True,
        normalize_hyphens: bool = True,
        **kwargs
    ):
        """
        Initialize the text normalizer.
        
        Args:
            lowercase: Whether to convert text to lowercase.
            remove_accents: Whether to remove accents from characters.
            normalize_whitespace: Whether to normalize whitespace.
            normalize_unicode: Whether to normalize Unicode characters.
            normalize_quotes: Whether to normalize quotes.
            normalize_hyphens: Whether to normalize hyphens.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
        self.normalize_quotes = normalize_quotes
        self.normalize_hyphens = normalize_hyphens
    
    async def process(self, text: str, **kwargs) -> str:
        """
        Normalize the input text.
        
        Args:
            text: Input text to normalize.
            **kwargs: Additional processing parameters.
            
        Returns:
            Normalized text.
        """
        if not text:
            return ""
        
        # Normalize Unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove accents
        if self.remove_accents:
            text = ''.join(c for c in unicodedata.normalize('NFD', text)
                          if unicodedata.category(c) != 'Mn')
        
        # Normalize quotes
        if self.normalize_quotes:
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'[\']', "'", text)
        
        # Normalize hyphens
        if self.normalize_hyphens:
            text = re.sub(r'[‐‑‒–—―]', '-', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        return text


class LLMTextNormalizer(BaseLLMProcessor):
    """
    A processor that uses an LLM to normalize text for consistency.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Normalize the following text for consistency:

{text}

Instructions:
- Standardize formatting and style
- Ensure consistent terminology
- Normalize date and number formats
- Standardize abbreviations and acronyms
- Maintain the original meaning and content
- Do not add or remove information

Normalized text:""",
        system_message="""You are a text normalization assistant. Your task is to normalize text for consistency while preserving its original meaning. Focus on standardizing formatting, terminology, and style without changing the content."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        **kwargs
    ):
        """
        Initialize the LLM text normalizer.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            **kwargs: Additional parameters for the base class.
        """
        prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        super().__init__(
            llm_manager=llm_manager,
            model_name=model_name,
            prompt_template=prompt_template,
            generation_options=generation_options,
            **kwargs
        )
    
    async def process(self, text: str, **kwargs) -> str:
        """
        Normalize the input text using an LLM.
        
        Args:
            text: Input text to normalize.
            **kwargs: Additional processing parameters.
            
        Returns:
            Normalized text.
        """
        if not text:
            return ""
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the normalized text
        normalized_text = await self._generate(prompt, system_message)
        
        return normalized_text
