

"""
Relationship extractor for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMExtractor
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class RelationshipExtractor(BaseLLMExtractor):
    """
    An extractor for identifying and extracting relationships between entities in text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Extract relationships between entities from the following text:

{text}

Identify entities and their relationships. For each relationship, provide:
- The source entity
- The target entity
- The type of relationship
- A confidence score from 0.0 to 1.0
- The sentence or context where the relationship was found

Format your response as a JSON object with a list of relationships:
```json
{{
  "relationships": [
    {{
      "source": {{"text": "John Smith", "type": "PERSON"}},
      "target": {{"text": "Acme Corp", "type": "ORGANIZATION"}},
      "relation": "works_for",
      "confidence": 0.9,
      "context": "John Smith is the CEO of Acme Corp."
    }},
    {{
      "source": {{"text": "Acme Corp", "type": "ORGANIZATION"}},
      "target": {{"text": "New York", "type": "LOCATION"}},
      "relation": "located_in",
      "confidence": 0.85,
      "context": "Acme Corp is headquartered in New York."
    }}
  ]
}}
```""",
        system_message="""You are a relationship extraction assistant. Your task is to identify and extract relationships between entities in text. Focus on accuracy and providing context for each relationship."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        relation_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the relationship extractor.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            relation_types: List of relationship types to extract.
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
        
        self.relation_types = relation_types or [
            "works_for", "located_in", "part_of", "affiliated_with",
            "created_by", "owned_by", "member_of", "related_to"
        ]
    
    async def extract(self, text: str, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships from the input text.
        
        Args:
            text: Input text to extract relationships from.
            **kwargs: Additional extraction parameters.
            
        Returns:
            Dictionary with a list of relationships.
        """
        if not text:
            return {"relationships": []}
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the extraction result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure the relationships key is present
        if "relationships" not in result:
            result["relationships"] = []
        
        return result
