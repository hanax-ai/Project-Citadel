
"""
Self-improvement loop for Project Citadel LangGraph integration.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_langgraph.feedback.response_evaluator import ResponseEvaluator


class SelfImprovementLoop:
    """
    Component for implementing self-improvement based on feedback.
    
    This class provides methods for refining responses and improving
    agent behavior based on evaluation results and feedback.
    """
    
    def __init__(
        self,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        evaluator: Optional[ResponseEvaluator] = None,
        max_iterations: int = 3,
        improvement_threshold: float = 0.5,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the self-improvement loop.
        
        Args:
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            evaluator: ResponseEvaluator instance to use. If None, a new one will be created.
            max_iterations: Maximum number of improvement iterations.
            improvement_threshold: Minimum score improvement required to continue iterations.
            logger: Logger instance.
        """
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.model_name = model_name
        self.evaluator = evaluator or ResponseEvaluator(
            ollama_gateway=self.ollama_gateway,
            model_name=self.model_name,
        )
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.logger = logger or get_logger("citadel.langgraph.feedback.self_improvement_loop")
        
        # Initialize the improvement templates
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """
        Initialize the templates for self-improvement.
        """
        # Template for refining responses
        self.refine_template = PromptTemplate.from_template("""
        You are an AI assistant tasked with improving a response to a user query.
        
        User Query: {query}
        
        Original Response: {response}
        
        Evaluation Feedback:
        {feedback}
        
        Improvement Suggestions:
        {suggestions}
        
        Please provide an improved response that addresses the feedback and suggestions.
        Focus on improving the aspects that received lower scores while maintaining the
        strengths of the original response.
        """)
        
        # Template for meta-prompt refinement
        self.meta_prompt_template = PromptTemplate.from_template("""
        You are an AI assistant tasked with improving the prompting strategy for a language model.
        
        Original Prompt: {prompt}
        
        User Query: {query}
        
        Response Generated: {response}
        
        Evaluation Feedback:
        {feedback}
        
        Based on this information, please suggest an improved prompt that would help
        the language model generate better responses for similar queries in the future.
        
        The improved prompt should:
        1. Address the weaknesses identified in the evaluation
        2. Maintain or enhance the strengths of the original response
        3. Be clear, specific, and actionable for the language model
        
        Improved Prompt:
        """)
    
    def refine_response(
        self,
        query: str,
        response: str,
        evaluation_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Refine a response based on evaluation results.
        
        Args:
            query: The user query.
            response: The response to refine.
            evaluation_results: Evaluation results. If None, the response will be evaluated.
            
        Returns:
            Dictionary containing the refined response and improvement metrics.
        """
        self.logger.info(f"Starting self-improvement loop for query: {query[:50]}...")
        
        # Evaluate the response if evaluation results are not provided
        if evaluation_results is None:
            evaluation_results = self.evaluator.evaluate(query, response)
        
        original_score = evaluation_results["overall_score"]
        self.logger.info(f"Original response score: {original_score:.2f}")
        
        # If the response already passes the threshold, no need to refine
        if evaluation_results["passed"]:
            self.logger.info("Response already meets quality threshold. No refinement needed.")
            return {
                "original_response": response,
                "refined_response": response,
                "original_score": original_score,
                "refined_score": original_score,
                "improvement": 0.0,
                "iterations": 0,
                "evaluation_history": [evaluation_results],
            }
        
        # Get improvement suggestions
        suggestions = self.evaluator.get_improvement_suggestions(evaluation_results)
        
        # Initialize variables for the refinement loop
        current_response = response
        current_score = original_score
        current_evaluation = evaluation_results
        evaluation_history = [evaluation_results]
        
        llm = self.ollama_gateway.get_llm(model=self.model_name)
        
        # Refinement loop
        for iteration in range(self.max_iterations):
            self.logger.info(f"Refinement iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate refined response
            refined_response = llm.invoke(
                self.refine_template.format(
                    query=query,
                    response=current_response,
                    feedback=current_evaluation["feedback"],
                    suggestions=suggestions,
                )
            )
            
            # Evaluate the refined response
            refined_evaluation = self.evaluator.evaluate(query, refined_response)
            evaluation_history.append(refined_evaluation)
            
            refined_score = refined_evaluation["overall_score"]
            improvement = refined_score - current_score
            
            self.logger.info(f"Refined response score: {refined_score:.2f} (improvement: {improvement:.2f})")
            
            # Check if the refinement improved the response
            if improvement < self.improvement_threshold:
                self.logger.info(f"Insufficient improvement ({improvement:.2f} < {self.improvement_threshold}). Stopping refinement.")
                break
            
            # Update current response and score
            current_response = refined_response
            current_score = refined_score
            current_evaluation = refined_evaluation
            
            # Get new improvement suggestions
            suggestions = self.evaluator.get_improvement_suggestions(current_evaluation)
            
            # Check if the response now passes the threshold
            if current_evaluation["passed"]:
                self.logger.info("Response now meets quality threshold. Stopping refinement.")
                break
        
        # Return the refinement results
        return {
            "original_response": response,
            "refined_response": current_response,
            "original_score": original_score,
            "refined_score": current_score,
            "improvement": current_score - original_score,
            "iterations": iteration + 1,
            "evaluation_history": evaluation_history,
        }
    
    def refine_prompt(
        self,
        prompt: str,
        query: str,
        response: str,
        evaluation_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Refine a prompt based on evaluation results.
        
        Args:
            prompt: The original prompt.
            query: The user query.
            response: The response generated using the prompt.
            evaluation_results: Evaluation results. If None, the response will be evaluated.
            
        Returns:
            Refined prompt.
        """
        self.logger.info(f"Refining prompt for query: {query[:50]}...")
        
        # Evaluate the response if evaluation results are not provided
        if evaluation_results is None:
            evaluation_results = self.evaluator.evaluate(query, response)
        
        llm = self.ollama_gateway.get_llm(model=self.model_name)
        
        # Generate refined prompt
        refined_prompt = llm.invoke(
            self.meta_prompt_template.format(
                prompt=prompt,
                query=query,
                response=response,
                feedback=evaluation_results["feedback"],
            )
        )
        
        self.logger.info("Prompt refinement complete.")
        
        return refined_prompt
    
    def create_improvement_callback(
        self,
        query_extractor: Callable[[Dict[str, Any]], str],
        response_extractor: Callable[[Dict[str, Any]], str],
        response_updater: Callable[[Dict[str, Any], str], Dict[str, Any]],
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Create a callback function for integrating self-improvement into a LangGraph workflow.
        
        Args:
            query_extractor: Function to extract the query from the state.
            response_extractor: Function to extract the response from the state.
            response_updater: Function to update the state with the refined response.
            
        Returns:
            Callback function for self-improvement.
        """
        def improvement_callback(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Callback function for self-improvement.
            
            Args:
                state: Current state.
                
            Returns:
                Updated state with refined response.
            """
            query = query_extractor(state)
            response = response_extractor(state)
            
            # Refine the response
            refinement_results = self.refine_response(query, response)
            
            # Update the state with the refined response
            updated_state = response_updater(state, refinement_results["refined_response"])
            
            # Add refinement metadata to the state
            if "metadata" not in updated_state:
                updated_state["metadata"] = {}
            
            updated_state["metadata"]["refinement"] = {
                "original_score": refinement_results["original_score"],
                "refined_score": refinement_results["refined_score"],
                "improvement": refinement_results["improvement"],
                "iterations": refinement_results["iterations"],
            }
            
            return updated_state
        
        return improvement_callback
