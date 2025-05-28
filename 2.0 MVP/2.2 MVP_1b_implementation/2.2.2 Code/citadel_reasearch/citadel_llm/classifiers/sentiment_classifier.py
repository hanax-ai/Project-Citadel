

"""
Sentiment classifier for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMClassifier
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class SentimentClassifier(BaseLLMClassifier):
    """
    A classifier for analyzing the sentiment of text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Analyze the sentiment of the following text:

{text}

Provide:
- The overall sentiment (positive, negative, or neutral)
- A sentiment score from -1.0 (very negative) to 1.0 (very positive)
- Confidence score from 0.0 to 1.0
- Key sentiment indicators (words, phrases, or expressions that indicate sentiment)
- A brief explanation of the sentiment analysis

Format your response as a JSON object:
```json
{{
  "sentiment": "positive",
  "score": 0.8,
  "confidence": 0.9,
  "indicators": ["excellent", "impressive", "highly recommend"],
  "explanation": "The text expresses strong approval and satisfaction."
}}
```""",
        system_message="""You are a sentiment analysis assistant. Your task is to analyze the sentiment expressed in text. Focus on identifying the emotional tone, attitude, and opinions expressed."""
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
        Initialize the sentiment classifier.
        
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
            output_format="json",
            **kwargs
        )
    
    async def classify(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze the sentiment of the input text.
        
        Args:
            text: Input text to analyze.
            **kwargs: Additional classification parameters.
            
        Returns:
            Dictionary with sentiment analysis results.
        """
        if not text:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "indicators": [],
                "explanation": "No text provided for sentiment analysis."
            }
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the classification result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure required keys are present
        required_keys = ["sentiment", "score", "confidence", "indicators", "explanation"]
        for key in required_keys:
            if key not in result:
                if key == "sentiment":
                    result[key] = "neutral"
                elif key == "score" or key == "confidence":
                    result[key] = 0.0
                elif key == "indicators":
                    result[key] = []
                else:
                    result[key] = ""
        
        return result
