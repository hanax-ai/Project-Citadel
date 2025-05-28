
"""
Keyword extractor for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMExtractor
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class KeywordExtractor(BaseLLMExtractor):
    """
    An extractor for identifying and extracting keywords and key phrases from text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Extract the most important keywords and key phrases from the following text:

{text}

Extract the following:
- KEYWORDS: Single words that represent important concepts
- KEYPHRASES: Multi-word phrases that represent important concepts
- TOPICS: Main topics or themes discussed in the text

For each keyword and key phrase, provide a relevance score from 0.0 to 1.0, where 1.0 is the most relevant.

Format your response as a JSON object with the following structure:
```json
{{
  "keywords": [
    {{"text": "AI", "relevance": 0.95}},
    {{"text": "machine learning", "relevance": 0.85}}
  ],
  "keyphrases": [
    {{"text": "natural language processing", "relevance": 0.9}},
    {{"text": "deep learning models", "relevance": 0.8}}
  ],
  "topics": [
    {{"text": "artificial intelligence", "relevance": 0.95}},
    {{"text": "technology trends", "relevance": 0.75}}
  ]
}}
```""",
        system_message="""You are a keyword extraction assistant. Your task is to identify and extract the most important keywords, key phrases, and topics from text. Focus on relevance and significance to the main themes of the text."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        max_keywords: int = 10,
        max_keyphrases: int = 10,
        max_topics: int = 5,
        **kwargs
    ):
        """
        Initialize the keyword extractor.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            max_keywords: Maximum number of keywords to extract.
            max_keyphrases: Maximum number of key phrases to extract.
            max_topics: Maximum number of topics to extract.
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
        
        self.max_keywords = max_keywords
        self.max_keyphrases = max_keyphrases
        self.max_topics = max_topics
    
    async def extract(self, text: str, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract keywords and key phrases from the input text.
        
        Args:
            text: Input text to extract keywords from.
            **kwargs: Additional extraction parameters.
            
        Returns:
            Dictionary with keywords, key phrases, and topics.
        """
        if not text:
            return {
                "keywords": [],
                "keyphrases": [],
                "topics": []
            }
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the extraction result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure all required keys are present
        for key in ["keywords", "keyphrases", "topics"]:
            if key not in result:
                result[key] = []
        
        # Limit the number of results
        result["keywords"] = result["keywords"][:self.max_keywords]
        result["keyphrases"] = result["keyphrases"][:self.max_keyphrases]
        result["topics"] = result["topics"][:self.max_topics]
        
        return result
