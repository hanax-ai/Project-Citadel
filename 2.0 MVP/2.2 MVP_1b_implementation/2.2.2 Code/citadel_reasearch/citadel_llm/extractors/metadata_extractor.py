

"""
Metadata extractor for the Citadel LLM package.
"""

from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseLLMExtractor
from citadel_llm.models import LLMManager, GenerationOptions
from citadel_llm.prompts import PromptTemplate


class MetadataExtractor(BaseLLMExtractor):
    """
    An extractor for identifying and extracting metadata from text.
    """
    
    DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
        template="""Extract metadata from the following text:

{text}

Extract the following metadata:
- title: The title or heading of the document
- author: The author(s) of the document
- date: The publication or creation date
- source: The source or publisher of the document
- language: The primary language of the document
- document_type: The type of document (article, report, etc.)
- main_topic: The main topic or subject of the document
- keywords: Key terms or concepts in the document (as an array)
- summary: A brief summary of the document (1-2 sentences)

Format your response as a JSON object:
```json
{{
  "title": "...",
  "author": "...",
  "date": "...",
  "source": "...",
  "language": "...",
  "document_type": "...",
  "main_topic": "...",
  "keywords": ["...", "...", "..."],
  "summary": "..."
}}
```

If any field cannot be determined from the text, use null for that field.""",
        system_message="""You are a metadata extraction assistant. Your task is to identify and extract metadata from documents. Focus on accuracy and extracting as much information as possible from the text."""
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
        Initialize the metadata extractor.
        
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
    
    async def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from the input text.
        
        Args:
            text: Input text to extract metadata from.
            **kwargs: Additional extraction parameters.
            
        Returns:
            Dictionary with extracted metadata.
        """
        if not text:
            return {
                "title": None,
                "author": None,
                "date": None,
                "source": None,
                "language": None,
                "document_type": None,
                "main_topic": None,
                "keywords": [],
                "summary": None
            }
        
        # Format the prompt
        prompt = self.prompt_template.format(text=text)
        system_message = self.prompt_template.get_system_message()
        
        # Generate the extraction result
        response = await self._generate(prompt, system_message)
        
        # Parse the JSON response
        result = self._parse_json_response(response)
        
        # Ensure all required keys are present
        required_keys = [
            "title", "author", "date", "source", "language",
            "document_type", "main_topic", "keywords", "summary"
        ]
        
        for key in required_keys:
            if key not in result:
                result[key] = None if key != "keywords" else []
        
        return result
    
    async def extract_from_pdf(self, pdf_content: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from PDF content.
        
        Args:
            pdf_content: PDF content dictionary from PDFProcessor.
            **kwargs: Additional extraction parameters.
            
        Returns:
            Dictionary with extracted metadata.
        """
        if not pdf_content:
            return {}
        
        # Extract metadata from PDF text
        text_metadata = await self.extract(pdf_content.get("text", ""), **kwargs)
        
        # Combine with existing PDF metadata
        pdf_metadata = pdf_content.get("metadata", {})
        
        # Merge metadata, preferring PDF metadata when available
        result = text_metadata.copy()
        
        # Map PDF metadata keys to our metadata keys
        mapping = {
            "title": "title",
            "author": "author",
            "creator": "author",
            "producer": "source",
            "subject": "main_topic",
            "keywords": "keywords"
        }
        
        for pdf_key, our_key in mapping.items():
            if pdf_key in pdf_metadata and pdf_metadata[pdf_key]:
                result[our_key] = pdf_metadata[pdf_key]
        
        return result
