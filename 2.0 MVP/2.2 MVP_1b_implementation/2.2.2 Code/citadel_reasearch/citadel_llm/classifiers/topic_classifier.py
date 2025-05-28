

"""
Topic classifier for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMClassifier
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class TopicClassifier(BaseLLMClassifier):
    """
    A classifier for categorizing text by topic.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Classify the following text into topics:

{text}

{topic_instruction}

For each identified topic, provide:
- The topic name
- A confidence score from 0.0 to 1.0
- A brief explanation of why this topic applies

Format your response as a JSON object:
```json
{{
  "primary_topic": "Technology",
  "topics": [
    {{
      "name": "Technology",
      "confidence": 0.9,
      "explanation": "The text discusses AI and machine learning technologies."
    }},
    {{
      "name": "Business",
      "confidence": 0.7,
      "explanation": "The text mentions business applications of AI."
    }}
  ]
}}
```""",
        system_message="""You are a topic classification assistant. Your task is to identify the topics discussed in a text. Focus on accuracy and providing clear explanations for your classifications."""
    )
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        model_name: str = "mistral:latest",
        prompt_template: Optional[PromptTemplate] = None,
        generation_options: Optional[GenerationOptions] = None,
        topics: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the topic classifier.
        
        Args:
            llm_manager: LLM manager instance.
            model_name: Name of the model to use.
            prompt_template: Prompt template to use.
            generation_options: Generation options.
            topics: List of predefined topics to classify into.
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
        
        self.topics = topics
    
    async def classify(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Classify the input text by topic.
        
        Args:
            text: Input text to classify.
            **kwargs: Additional classification parameters.
            
        Returns:
            Dictionary with classification results.
        """
        if not text:
            return {
                "primary_topic": None,
                "topics": []
            }
        
        # Determine topic instruction based on whether topics are predefined
        if self.topics:
            topic_instruction = f"Classify the text into one or more of the following topics: {', '.join(self.topics)}."
        else:
            topic_instruction = "Identify the most relevant topics discussed in the text."
        
        # Format the prompt
        prompt = self.prompt_template.format(
            text=text,
            topic_instruction=topic_instruction
        )
        system_message = self.prompt_template.get_system_message()
        
        # Generate the classification result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure required keys are present
        if "primary_topic" not in result:
            if "topics" in result and result["topics"]:
                # Use the first topic as the primary topic
                result["primary_topic"] = result["topics"][0]["name"]
            else:
                result["primary_topic"] = None
        
        if "topics" not in result:
            result["topics"] = []
        
        return result
