
"""
Prompt templates and system messages for the Citadel LLM package.

This module provides prompt templates and system messages for different use cases.
"""

from typing import Dict, Any, Optional


class PromptTemplate:
    """A template for generating prompts."""
    
    def __init__(self, template: str, system_message: Optional[str] = None):
        """
        Initialize the prompt template.
        
        Args:
            template: The prompt template string.
            system_message: The system message to use with this template.
        """
        self.template = template
        self.system_message = system_message
    
    def format(self, **kwargs: Any) -> str:
        """
        Format the template with the given arguments.
        
        Args:
            **kwargs: Arguments to format the template with.
            
        Returns:
            Formatted prompt.
        """
        return self.template.format(**kwargs)
    
    def get_system_message(self) -> Optional[str]:
        """
        Get the system message for this template.
        
        Returns:
            System message or None if not set.
        """
        return self.system_message


# System messages for different use cases
SYSTEM_MESSAGES = {
    "default": """You are a helpful AI assistant.""",
    
    "code": """You are a helpful AI assistant specialized in programming and software development.
Your task is to provide accurate, efficient, and well-documented code solutions.
Always explain your code and provide context for your solutions.""",
    
    "analysis": """You are a helpful AI assistant specialized in data analysis and interpretation.
Your task is to provide insightful analysis, identify patterns, and draw meaningful conclusions from data.
Always explain your reasoning and provide context for your analysis.""",
    
    "research": """You are a helpful AI assistant specialized in research and information gathering.
Your task is to provide comprehensive, accurate, and well-sourced information on various topics.
Always cite your sources and provide context for your information.""",
    
    "creative": """You are a helpful AI assistant specialized in creative writing and content generation.
Your task is to provide engaging, original, and high-quality content based on the given prompts.
Always be creative, but maintain coherence and relevance to the topic.""",
}


# Prompt templates for different use cases
PROMPT_TEMPLATES = {
    "code_generation": PromptTemplate(
        template="""Write a {language} function to {task}. 
The function should be named {function_name} and take the following parameters: {parameters}.
{additional_context}""",
        system_message=SYSTEM_MESSAGES["code"]
    ),
    
    "code_explanation": PromptTemplate(
        template="""Explain the following {language} code:
```{language}
{code}
```
{additional_context}""",
        system_message=SYSTEM_MESSAGES["code"]
    ),
    
    "code_review": PromptTemplate(
        template="""Review the following {language} code and suggest improvements:
```{language}
{code}
```
{additional_context}""",
        system_message=SYSTEM_MESSAGES["code"]
    ),
    
    "data_analysis": PromptTemplate(
        template="""Analyze the following data and provide insights:
{data}
{additional_context}""",
        system_message=SYSTEM_MESSAGES["analysis"]
    ),
    
    "research_query": PromptTemplate(
        template="""Research the following topic and provide a comprehensive summary:
Topic: {topic}
{additional_context}""",
        system_message=SYSTEM_MESSAGES["research"]
    ),
    
    "creative_writing": PromptTemplate(
        template="""Write a {content_type} about {topic} with the following characteristics:
Style: {style}
Tone: {tone}
Length: {length}
{additional_context}""",
        system_message=SYSTEM_MESSAGES["creative"]
    ),
    
    "general_query": PromptTemplate(
        template="{query}",
        system_message=SYSTEM_MESSAGES["default"]
    ),
}


def get_prompt_template(template_name: str) -> PromptTemplate:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template.
        
    Returns:
        Prompt template.
        
    Raises:
        KeyError: If the template is not found.
    """
    return PROMPT_TEMPLATES[template_name]


def get_system_message(message_name: str) -> str:
    """
    Get a system message by name.
    
    Args:
        message_name: Name of the system message.
        
    Returns:
        System message.
        
    Raises:
        KeyError: If the message is not found.
    """
    return SYSTEM_MESSAGES[message_name]


def format_prompt(template_name: str, **kwargs: Any) -> Dict[str, str]:
    """
    Format a prompt template with the given arguments.
    
    Args:
        template_name: Name of the template.
        **kwargs: Arguments to format the template with.
        
    Returns:
        Dictionary with formatted prompt and system message.
        
    Raises:
        KeyError: If the template is not found.
    """
    template = get_prompt_template(template_name)
    return {
        "prompt": template.format(**kwargs),
        "system_message": template.get_system_message()
    }
