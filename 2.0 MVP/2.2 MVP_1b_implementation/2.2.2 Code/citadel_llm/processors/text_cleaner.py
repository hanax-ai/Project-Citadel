
"""
Text cleaner for the Citadel LLM package.
"""

import re
import string
from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseProcessor, BaseLLMProcessor
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class TextCleaner(BaseProcessor):
    """
    A processor for cleaning text by removing unwanted characters,
    HTML tags, extra whitespace, etc.
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        lowercase: bool = False,
        custom_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the text cleaner.
        
        Args:
            remove_html: Whether to remove HTML tags.
            remove_urls: Whether to remove URLs.
            remove_emails: Whether to remove email addresses.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            remove_extra_whitespace: Whether to remove extra whitespace.
            lowercase: Whether to convert text to lowercase.
            custom_patterns: Custom regex patterns to remove.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.lowercase = lowercase
        self.custom_patterns = custom_patterns or []
        
        # Compile regex patterns
        self.patterns = []
        
        if self.remove_html:
            self.patterns.append((re.compile(r'<.*?>'), ' '))
        
        if self.remove_urls:
            self.patterns.append((re.compile(r'https?://\S+|www\.\S+'), ' '))
        
        if self.remove_emails:
            self.patterns.append((re.compile(r'\S+@\S+'), ' '))
        
        for pattern in self.custom_patterns:
            self.patterns.append((re.compile(pattern), ' '))
    
    async def process(self, text: str, **kwargs) -> str:
        """
        Clean the input text.
        
        Args:
            text: Input text to clean.
            **kwargs: Additional processing parameters.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Apply regex patterns
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        return text


class LLMTextCleaner(BaseLLMProcessor):
    """
    A processor that uses an LLM to clean and improve text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Clean and improve the following text while preserving its meaning:

{text}

Instructions:
- Fix grammatical errors
- Fix spelling mistakes
- Remove redundant information
- Improve clarity and readability
- Preserve the original meaning and tone
- Maintain the original structure (paragraphs, bullet points, etc.)
- Do not add new information

Cleaned text:""",
        system_message="""You are a text cleaning assistant. Your task is to clean and improve text while preserving its original meaning and structure. Focus on fixing grammatical errors, spelling mistakes, and improving clarity without adding new information."""
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
        Initialize the LLM text cleaner.
        
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
        Clean the input text using an LLM.
        
        Args:
            text: Input text to clean.
            **kwargs: Additional processing parameters.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the cleaned text
        cleaned_text = await self._generate(prompt, system_message)
        
        return cleaned_text
