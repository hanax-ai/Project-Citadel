
"""
Base evaluator class for Project Citadel LangGraph integration.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluator components in Project Citadel.
    
    This class provides a common interface for all evaluator components and
    handles integration with LangChain's evaluation components.
    """
    
    def __init__(
        self,
        evaluator: Optional[Runnable] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        criteria: Optional[List[str]] = None,
        evaluation_template: Optional[Union[str, PromptTemplate]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the evaluator component.
        
        Args:
            evaluator: LangChain evaluator component to use. If None, a new one will be created.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            criteria: List of criteria to evaluate against.
            evaluation_template: Template for evaluation prompts.
            logger: Logger instance.
        """
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.model_name = model_name
        self.criteria = criteria or ["correctness", "relevance", "coherence", "helpfulness"]
        self.logger = logger or get_logger(f"citadel.langgraph.feedback.{self.__class__.__name__.lower()}")
        
        # The underlying LangChain evaluator component
        self._evaluator = evaluator
        
        # Set up the evaluation template
        if evaluation_template is None:
            self._evaluation_template = self._get_default_evaluation_template()
        elif isinstance(evaluation_template, str):
            self._evaluation_template = PromptTemplate.from_template(evaluation_template)
        else:
            self._evaluation_template = evaluation_template
    
    def _get_default_evaluation_template(self) -> PromptTemplate:
        """
        Get the default evaluation template.
        
        Returns:
            Default evaluation template.
        """
        template = """
        You are an expert evaluator tasked with assessing the quality of an AI assistant's response.
        
        User Query: {query}
        
        AI Response: {response}
        
        Please evaluate the response based on the following criteria:
        {criteria}
        
        For each criterion, provide a score from 1-10 and a brief explanation.
        Then provide an overall score and summary of the evaluation.
        """
        return PromptTemplate.from_template(template)
    
    @abstractmethod
    def evaluate(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a response.
        
        Args:
            query: The user query.
            response: The response to evaluate.
            **kwargs: Additional parameters.
            
        Returns:
            Evaluation results.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the evaluator to disk.
        
        Args:
            path: Path to save to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the evaluator
        with open(path, "wb") as f:
            pickle.dump(self._evaluator, f)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseEvaluator":
        """
        Load the evaluator from disk.
        
        Args:
            path: Path to load from.
            **kwargs: Additional parameters.
            
        Returns:
            Loaded evaluator.
        """
        # Load the evaluator
        with open(path, "rb") as f:
            evaluator = pickle.load(f)
        
        # Create a new instance
        return cls(evaluator=evaluator, **kwargs)
    
    def to_graph_node(self) -> Dict[str, Any]:
        """
        Convert the evaluator to a graph node.
        
        Returns:
            Dictionary representation of the evaluator as a graph node.
        """
        return {
            "type": "evaluator",
            "class": self.__class__.__name__,
            "criteria": self.criteria,
        }
