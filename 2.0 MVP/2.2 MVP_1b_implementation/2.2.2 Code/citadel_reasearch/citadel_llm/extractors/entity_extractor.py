
"""
Entity extractor for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMExtractor
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class EntityExtractor(BaseLLMExtractor):
    """
    An extractor for identifying and extracting named entities from text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Extract all named entities from the following text:

{text}

Extract the following entity types:
- PERSON: Names of people
- ORGANIZATION: Names of companies, agencies, institutions
- LOCATION: Names of locations, cities, countries
- DATE: Dates or periods
- EVENT: Named events
- PRODUCT: Names of products
- WORK: Titles of books, songs, etc.

Format your response as a JSON object with entity types as keys and lists of entities as values. Include the position (start and end character indices) of each entity in the text.

Example format:
```json
{{
  "PERSON": [
    {{"text": "John Smith", "start": 10, "end": 20}},
    {{"text": "Jane Doe", "start": 45, "end": 53}}
  ],
  "ORGANIZATION": [
    {{"text": "Acme Corp", "start": 100, "end": 109}}
  ]
}}
```""",
        system_message="""You are an entity extraction assistant. Your task is to identify and extract named entities from text. Focus on accuracy and completeness. Extract entities exactly as they appear in the text, including their positions."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        entity_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the entity extractor.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            entity_types: List of entity types to extract.
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
        
        self.entity_types = entity_types or [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "PRODUCT", "WORK"
        ]
    
    async def extract(self, text: str, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from the input text.
        
        Args:
            text: Input text to extract entities from.
            **kwargs: Additional extraction parameters.
            
        Returns:
            Dictionary with entity types as keys and lists of entities as values.
        """
        if not text:
            return {entity_type: [] for entity_type in self.entity_types}
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the extraction result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure all entity types are present in the result
        for entity_type in self.entity_types:
            if entity_type not in result:
                result[entity_type] = []
        
        return result
