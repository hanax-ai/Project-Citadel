
"""
Multi-level summarizer for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Union

from .base import BaseSummarizer
from .abstractive_summarizer import AbstractiveSummarizer
from citadel_llm.models import LLMManager, GenerationOptions


class MultiLevelSummarizer(BaseSummarizer):
    """
    A summarizer that generates summaries at multiple levels of detail.
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        generation_options: Optional[GenerationOptions] = None,
        **kwargs
    ):
        """
        Initialize the multi-level summarizer.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            generation_options: Generation options.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        # Create summarizers for different levels
        self.summarizers = {
            "headline": AbstractiveSummarizer(
                llm_manager=llm_manager,
                model_name=model_name,
                generation_options=generation_options,
                summary_type="short",
                max_words=15,
                logger=self.logger
            ),
            "short": AbstractiveSummarizer(
                llm_manager=llm_manager,
                model_name=model_name,
                generation_options=generation_options,
                summary_type="short",
                max_words=50,
                logger=self.logger
            ),
            "medium": AbstractiveSummarizer(
                llm_manager=llm_manager,
                model_name=model_name,
                generation_options=generation_options,
                summary_type="medium",
                max_words=150,
                logger=self.logger
            ),
            "long": AbstractiveSummarizer(
                llm_manager=llm_manager,
                model_name=model_name,
                generation_options=generation_options,
                summary_type="long",
                max_words=300,
                logger=self.logger
            )
        }
    
    async def summarize(
        self,
        text: str,
        level: str = "medium",
        **kwargs
    ) -> str:
        """
        Generate a summary at the specified level of detail.
        
        Args:
            text: Input text to summarize.
            level: Level of detail ("headline", "short", "medium", or "long").
            **kwargs: Additional summarization parameters.
            
        Returns:
            Summarized text.
        """
        if not text:
            return ""
        
        # Use the appropriate summarizer for the requested level
        if level in self.summarizers:
            return await self.summarizers[level].summarize(text, **kwargs)
        else:
            self.logger.warning(f"Unknown summary level: {level}, using 'medium' instead")
            return await self.summarizers["medium"].summarize(text, **kwargs)
    
    async def summarize_all_levels(
        self,
        text: str,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate summaries at all levels of detail.
        
        Args:
            text: Input text to summarize.
            **kwargs: Additional summarization parameters.
            
        Returns:
            Dictionary with summaries at all levels.
        """
        if not text:
            return {level: "" for level in self.summarizers}
        
        # Generate summaries at all levels
        results = {}
        
        for level, summarizer in self.summarizers.items():
            results[level] = await summarizer.summarize(text, **kwargs)
        
        return results
    
    async def summarize_pdf(
        self,
        pdf_content: Dict[str, Any],
        level: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a summary of PDF content.
        
        Args:
            pdf_content: PDF content dictionary from PDFProcessor.
            level: Level of detail ("headline", "short", "medium", or "long").
            **kwargs: Additional summarization parameters.
            
        Returns:
            Dictionary with the original PDF content and added summary.
        """
        if not pdf_content or "text" not in pdf_content:
            return pdf_content
        
        # Create a copy of the PDF content
        result = pdf_content.copy()
        
        # Generate the summary
        summary = await self.summarize(pdf_content["text"], level=level, **kwargs)
        
        # Add the summary to the result
        if "summaries" not in result:
            result["summaries"] = {}
        
        result["summaries"][level] = summary
        
        return result
    
    async def summarize_pdf_all_levels(
        self,
        pdf_content: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate summaries of PDF content at all levels of detail.
        
        Args:
            pdf_content: PDF content dictionary from PDFProcessor.
            **kwargs: Additional summarization parameters.
            
        Returns:
            Dictionary with the original PDF content and added summaries.
        """
        if not pdf_content or "text" not in pdf_content:
            return pdf_content
        
        # Create a copy of the PDF content
        result = pdf_content.copy()
        
        # Generate summaries at all levels
        summaries = await self.summarize_all_levels(pdf_content["text"], **kwargs)
        
        # Add the summaries to the result
        result["summaries"] = summaries
        
        return result
