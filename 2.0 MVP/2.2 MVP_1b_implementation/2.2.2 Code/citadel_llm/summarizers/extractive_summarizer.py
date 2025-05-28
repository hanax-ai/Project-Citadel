

"""
Extractive summarizer for the Citadel LLM package.
"""

import re
from typing import Any, Dict, List, Optional, Set, Union, Tuple

from .base import BaseLLMSummarizer
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class ExtractiveSummarizer(BaseLLMSummarizer):
    """
    A summarizer that generates extractive summaries by selecting
    important sentences from the original text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Extract the most important sentences from the following text to create a summary:

{text}

Instructions:
- Select {num_sentences} sentences that best represent the main points
- Do not modify the selected sentences
- Maintain the original order of the sentences
- Focus on sentences that contain key information

Format your response as a JSON array of sentence indices (0-based):
```json
[3, 7, 12, 15, 20]
```

These indices should correspond to the most important sentences in the text.""",
        system_message="""You are an extractive summarization assistant. Your task is to identify and extract the most important sentences from text to create a summary. Focus on sentences that contain key information, main points, and essential context."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        num_sentences: int = 5,
        **kwargs
    ):
        """
        Initialize the extractive summarizer.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            num_sentences: Number of sentences to extract.
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
        
        self.num_sentences = num_sentences
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text.
            
        Returns:
            List of sentences.
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def summarize(self, text: str, **kwargs) -> str:
        """
        Generate an extractive summary of the input text.
        
        Args:
            text: Input text to summarize.
            **kwargs: Additional summarization parameters.
            
        Returns:
            Summarized text.
        """
        if not text:
            return ""
        
        # Override num_sentences if provided in kwargs
        num_sentences = kwargs.get("num_sentences", self.num_sentences)
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # If there are fewer sentences than requested, return the original text
        if len(sentences) <= num_sentences:
            return text
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text, num_sentences=num_sentences)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the sentence indices
        response = await self._generate(prompt, system_message)
        
        # Parse the response to get sentence indices
        try:
            # Extract JSON array from response
            import json
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                indices_str = match.group(0)
                indices = json.loads(indices_str)
                
                # Validate indices
                valid_indices = [i for i in indices if 0 <= i < len(sentences)]
                valid_indices.sort()  # Maintain original order
                
                # Extract the selected sentences
                selected_sentences = [sentences[i] for i in valid_indices]
                
                # Join the selected sentences
                return " ".join(selected_sentences)
            else:
                self.logger.warning("No sentence indices found in response")
                return self._fallback_summarize(sentences, num_sentences)
        except Exception as e:
            self.logger.error(f"Error parsing sentence indices: {str(e)}")
            return self._fallback_summarize(sentences, num_sentences)
    
    def _fallback_summarize(self, sentences: List[str], num_sentences: int) -> str:
        """
        Fallback method for summarization when LLM extraction fails.
        
        Args:
            sentences: List of sentences.
            num_sentences: Number of sentences to extract.
            
        Returns:
            Summarized text.
        """
        # Simple fallback: take the first sentence and evenly spaced sentences
        if not sentences:
            return ""
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # Always include the first sentence
        selected_indices = [0]
        
        # Add evenly spaced sentences
        if num_sentences > 1:
            step = len(sentences) / (num_sentences - 1)
            for i in range(1, num_sentences - 1):
                index = min(int(i * step), len(sentences) - 1)
                if index not in selected_indices:
                    selected_indices.append(index)
            
            # Always include the last sentence
            if len(sentences) - 1 not in selected_indices:
                selected_indices.append(len(sentences) - 1)
        
        # Sort indices to maintain original order
        selected_indices.sort()
        
        # Extract the selected sentences
        selected_sentences = [sentences[i] for i in selected_indices]
        
        return " ".join(selected_sentences)
