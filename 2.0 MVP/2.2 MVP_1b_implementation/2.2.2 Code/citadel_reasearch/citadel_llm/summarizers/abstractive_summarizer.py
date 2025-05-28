

"""
Abstractive summarizer for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Union

from .base import BaseLLMSummarizer
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class AbstractiveSummarizer(BaseLLMSummarizer):
    """
    A summarizer that generates abstractive summaries using an LLM.
    """
    
    DEFAULT_PROMPT_TEMPLATES = {
        "short": PromptTemplate(
            template="""Create a concise summary of the following text in {max_words} words or less:

{text}

Summary:""",
            system_message="""You are a summarization assistant. Your task is to create concise, accurate summaries of text. Focus on capturing the main points and key information while staying within the specified length limit."""
        ),
        "medium": PromptTemplate(
            template="""Create a comprehensive summary of the following text in {max_words} words or less:

{text}

Summary:""",
            system_message="""You are a summarization assistant. Your task is to create comprehensive, accurate summaries of text. Focus on capturing the main points, key details, and important context while staying within the specified length limit."""
        ),
        "long": PromptTemplate(
            template="""Create a detailed summary of the following text in {max_words} words or less:

{text}

Include:
- Main points and arguments
- Key supporting details
- Important conclusions
- Significant context

Summary:""",
            system_message="""You are a summarization assistant. Your task is to create detailed, accurate summaries of text. Focus on capturing the main points, supporting details, conclusions, and context while staying within the specified length limit."""
        )
    }
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        summary_type: str = "medium",
        max_words: int = 200,
        **kwargs
    ):
        """
        Initialize the abstractive summarizer.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            summary_type: Type of summary ("short", "medium", or "long").
            max_words: Maximum number of words in the summary.
            **kwargs: Additional parameters for the base class.
        """
        # Select the appropriate prompt template based on summary type
        if prompt_template is None:
            if summary_type in self.DEFAULT_PROMPT_TEMPLATES:
                prompt_template = self.DEFAULT_PROMPT_TEMPLATES[summary_type]
            else:
                prompt_template = self.DEFAULT_PROMPT_TEMPLATES["medium"]
        
        super().__init__(
            llm_manager=llm_manager,
            model_name=model_name,
            prompt_template=prompt_template,
            generation_options=generation_options,
            **kwargs
        )
        
        self.summary_type = summary_type
        self.max_words = max_words
    
    async def summarize(self, text: str, **kwargs) -> str:
        """
        Generate an abstractive summary of the input text.
        
        Args:
            text: Input text to summarize.
            **kwargs: Additional summarization parameters.
            
        Returns:
            Summarized text.
        """
        if not text:
            return ""
        
        # Override max_words if provided in kwargs
        max_words = kwargs.get("max_words", self.max_words)
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text, max_words=max_words)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the summary
        summary = await self._generate(prompt, system_message)
        
        return summary
    
    async def summarize_chunks(
        self,
        chunks: List[str],
        hierarchical: bool = True,
        **kwargs
    ) -> str:
        """
        Summarize a list of text chunks.
        
        Args:
            chunks: List of text chunks to summarize.
            hierarchical: Whether to use hierarchical summarization.
            **kwargs: Additional summarization parameters.
            
        Returns:
            Summarized text.
        """
        if not chunks:
            return ""
        
        # If there's only one chunk, summarize it directly
        if len(chunks) == 1:
            return await self.summarize(chunks[0], **kwargs)
        
        # For hierarchical summarization
        if hierarchical:
            # First, summarize each chunk individually
            chunk_summaries = []
            for chunk in chunks:
                # Use a smaller max_words for individual chunks
                chunk_max_words = min(100, self.max_words // 2)
                summary = await self.summarize(chunk, max_words=chunk_max_words)
                chunk_summaries.append(summary)
            
            # Then, summarize the combined summaries
            combined_text = "\n\n".join(chunk_summaries)
            return await self.summarize(combined_text, **kwargs)
        
        # For non-hierarchical summarization, combine chunks and summarize
        combined_text = "\n\n".join(chunks)
        return await self.summarize(combined_text, **kwargs)
