

"""
Intent classifier for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMClassifier
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class IntentClassifier(BaseLLMClassifier):
    """
    A classifier for identifying the intent of text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Identify the intent of the following text:

{text}

{intent_instruction}

For each identified intent, provide:
- The intent name
- A confidence score from 0.0 to 1.0
- A brief explanation of why this intent applies

Format your response as a JSON object:
```json
{{
  "primary_intent": "information_request",
  "intents": [
    {{
      "name": "information_request",
      "confidence": 0.9,
      "explanation": "The text is asking for specific information about a topic."
    }},
    {{
      "name": "clarification",
      "confidence": 0.6,
      "explanation": "The text is also seeking clarification on a previous point."
    }}
  ]
}}
```""",
        system_message="""You are an intent classification assistant. Your task is to identify the purpose or intent behind text. Focus on understanding what the author is trying to achieve or communicate."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        intents: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the intent classifier.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            intents: List of predefined intents to classify into.
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
        
        self.intents = intents or [
            "information_request",
            "opinion_expression",
            "complaint",
            "suggestion",
            "instruction",
            "clarification",
            "agreement",
            "disagreement",
            "gratitude",
            "apology"
        ]
    
    async def classify(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Classify the intent of the input text.
        
        Args:
            text: Input text to classify.
            **kwargs: Additional classification parameters.
            
        Returns:
            Dictionary with intent classification results.
        """
        if not text:
            return {
                "primary_intent": None,
                "intents": []
            }
        
        # Determine intent instruction based on whether intents are predefined
        intent_instruction = f"Classify the text into one or more of the following intents: {', '.join(self.intents)}."
        
        # Format the prompt
        prompt = self.prompt_template.format(
            text=text,
            intent_instruction=intent_instruction
        )
        system_message = self.prompt_template.get_system_message()
        
        # Generate the classification result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure required keys are present
        if "primary_intent" not in result:
            if "intents" in result and result["intents"]:
                # Use the first intent as the primary intent
                result["primary_intent"] = result["intents"][0]["name"]
            else:
                result["primary_intent"] = None
        
        if "intents" not in result:
            result["intents"] = []
        
        return result
