

"""
Content type classifier for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMClassifier
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class ContentTypeClassifier(BaseLLMClassifier):
    """
    A classifier for categorizing text by content type.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Classify the following text by content type:

{text}

{type_instruction}

For each identified content type, provide:
- The content type name
- A confidence score from 0.0 to 1.0
- A brief explanation of why this content type applies

Format your response as a JSON object:
```json
{{
  "primary_type": "article",
  "types": [
    {{
      "name": "article",
      "confidence": 0.9,
      "explanation": "The text has a structured format with paragraphs, headings, and a cohesive narrative."
    }},
    {{
      "name": "technical_documentation",
      "confidence": 0.7,
      "explanation": "The text contains technical details and explanations."
    }}
  ]
}}
```""",
        system_message="""You are a content type classification assistant. Your task is to identify the type or genre of content. Focus on structural and stylistic features that indicate the content type."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        content_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the content type classifier.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            content_types: List of predefined content types to classify into.
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
        
        self.content_types = content_types or [
            "article",
            "blog_post",
            "news",
            "academic_paper",
            "technical_documentation",
            "tutorial",
            "review",
            "opinion_piece",
            "social_media_post",
            "email",
            "legal_document",
            "creative_writing",
            "advertisement",
            "report"
        ]
    
    async def classify(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Classify the content type of the input text.
        
        Args:
            text: Input text to classify.
            **kwargs: Additional classification parameters.
            
        Returns:
            Dictionary with content type classification results.
        """
        if not text:
            return {
                "primary_type": None,
                "types": []
            }
        
        # Determine type instruction based on whether content types are predefined
        type_instruction = f"Classify the text into one or more of the following content types: {', '.join(self.content_types)}."
        
        # Format the prompt
        prompt = self.prompt_template.format(
            text=text,
            type_instruction=type_instruction
        )
        system_message = self.prompt_template.get_system_message()
        
        # Generate the classification result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure required keys are present
        if "primary_type" not in result:
            if "types" in result and result["types"]:
                # Use the first type as the primary type
                result["primary_type"] = result["types"][0]["name"]
            else:
                result["primary_type"] = None
        
        if "types" not in result:
            result["types"] = []
        
        return result
