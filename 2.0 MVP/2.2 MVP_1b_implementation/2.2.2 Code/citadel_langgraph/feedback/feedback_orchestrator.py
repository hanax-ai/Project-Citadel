
"""
Feedback orchestrator for Project Citadel LangGraph integration.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_langgraph.feedback.response_evaluator import ResponseEvaluator
from citadel_langgraph.feedback.feedback_collector import FeedbackCollector
from citadel_langgraph.feedback.self_improvement_loop import SelfImprovementLoop


class FeedbackOrchestrator:
    """
    Component for coordinating the feedback process.
    
    This class orchestrates the feedback process, including evaluation,
    human feedback collection, and self-improvement.
    """
    
    def __init__(
        self,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        evaluator: Optional[ResponseEvaluator] = None,
        feedback_collector: Optional[FeedbackCollector] = None,
        self_improvement_loop: Optional[SelfImprovementLoop] = None,
        auto_improve: bool = True,
        collect_human_feedback: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the feedback orchestrator.
        
        Args:
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            evaluator: ResponseEvaluator instance to use. If None, a new one will be created.
            feedback_collector: FeedbackCollector instance to use. If None, a new one will be created.
            self_improvement_loop: SelfImprovementLoop instance to use. If None, a new one will be created.
            auto_improve: Whether to automatically improve responses.
            collect_human_feedback: Whether to collect human feedback.
            logger: Logger instance.
        """
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.model_name = model_name
        self.auto_improve = auto_improve
        self.collect_human_feedback = collect_human_feedback
        self.logger = logger or get_logger("citadel.langgraph.feedback.feedback_orchestrator")
        
        # Initialize components
        self.evaluator = evaluator or ResponseEvaluator(
            ollama_gateway=self.ollama_gateway,
            model_name=self.model_name,
        )
        
        self.feedback_collector = feedback_collector or FeedbackCollector(
            ollama_gateway=self.ollama_gateway,
            model_name=self.model_name,
        )
        
        self.self_improvement_loop = self_improvement_loop or SelfImprovementLoop(
            ollama_gateway=self.ollama_gateway,
            model_name=self.model_name,
            evaluator=self.evaluator,
        )
    
    def process_response(
        self,
        query: str,
        response: str,
        human_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a response through the feedback pipeline.
        
        Args:
            query: The user query.
            response: The response to process.
            human_feedback: Optional human feedback.
            
        Returns:
            Processing results.
        """
        self.logger.info(f"Processing response for query: {query[:50]}...")
        
        # Step 1: Evaluate the response
        evaluation_results = self.evaluator.evaluate(query, response)
        
        # Step 2: Collect human feedback if provided
        feedback_entry = None
        if human_feedback and self.collect_human_feedback:
            feedback_entry = self.feedback_collector.collect_feedback(
                query=query,
                response=response,
                feedback=human_feedback.get("feedback", ""),
                rating=human_feedback.get("rating"),
                feedback_type="human",
                metadata={"evaluation_results": evaluation_results},
            )
        
        # Step 3: Improve the response if needed and auto_improve is enabled
        refinement_results = None
        if self.auto_improve and not evaluation_results["passed"]:
            refinement_results = self.self_improvement_loop.refine_response(
                query=query,
                response=response,
                evaluation_results=evaluation_results,
            )
            
            # Collect automated feedback on the refinement
            if self.collect_human_feedback:
                self.feedback_collector.collect_feedback(
                    query=query,
                    response=refinement_results["refined_response"],
                    feedback=f"Auto-refined response. Original score: {refinement_results['original_score']:.2f}, Refined score: {refinement_results['refined_score']:.2f}",
                    rating=int(refinement_results["refined_score"] / 2),  # Convert 1-10 scale to 1-5
                    feedback_type="automated",
                    metadata={"refinement_results": refinement_results},
                )
        
        # Prepare the results
        results = {
            "query": query,
            "original_response": response,
            "evaluation_results": evaluation_results,
            "human_feedback": feedback_entry,
            "refinement_results": refinement_results,
            "final_response": refinement_results["refined_response"] if refinement_results else response,
        }
        
        self.logger.info("Response processing complete.")
        
        return results
    
    def create_feedback_node(
        self,
        query_extractor: Callable[[Dict[str, Any]], str],
        response_extractor: Callable[[Dict[str, Any]], str],
        response_updater: Callable[[Dict[str, Any], str], Dict[str, Any]],
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Create a feedback node for a LangGraph workflow.
        
        Args:
            query_extractor: Function to extract the query from the state.
            response_extractor: Function to extract the response from the state.
            response_updater: Function to update the state with the processed response.
            
        Returns:
            Feedback node function.
        """
        def feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Feedback node function.
            
            Args:
                state: Current state.
                
            Returns:
                Updated state with processed response.
            """
            query = query_extractor(state)
            response = response_extractor(state)
            
            # Process the response
            results = self.process_response(query, response)
            
            # Update the state with the processed response
            updated_state = response_updater(state, results["final_response"])
            
            # Add feedback metadata to the state
            if "metadata" not in updated_state:
                updated_state["metadata"] = {}
            
            updated_state["metadata"]["feedback"] = {
                "evaluation_score": results["evaluation_results"]["overall_score"],
                "passed_evaluation": results["evaluation_results"]["passed"],
                "has_human_feedback": results["human_feedback"] is not None,
                "was_refined": results["refinement_results"] is not None,
            }
            
            if results["refinement_results"]:
                updated_state["metadata"]["feedback"]["refinement"] = {
                    "original_score": results["refinement_results"]["original_score"],
                    "refined_score": results["refinement_results"]["refined_score"],
                    "improvement": results["refinement_results"]["improvement"],
                    "iterations": results["refinement_results"]["iterations"],
                }
            
            return updated_state
        
        return feedback_node
    
    def create_human_feedback_collector(
        self,
        query_extractor: Callable[[Dict[str, Any]], str],
        response_extractor: Callable[[Dict[str, Any]], str],
    ) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
        """
        Create a human feedback collector for a LangGraph workflow.
        
        Args:
            query_extractor: Function to extract the query from the state.
            response_extractor: Function to extract the response from the state.
            
        Returns:
            Human feedback collector function.
        """
        def human_feedback_collector(
            state: Dict[str, Any],
            human_feedback: Dict[str, Any],
        ) -> Dict[str, Any]:
            """
            Human feedback collector function.
            
            Args:
                state: Current state.
                human_feedback: Human feedback.
                
            Returns:
                Updated state with human feedback.
            """
            query = query_extractor(state)
            response = response_extractor(state)
            
            # Collect human feedback
            feedback_entry = self.feedback_collector.collect_feedback(
                query=query,
                response=response,
                feedback=human_feedback.get("feedback", ""),
                rating=human_feedback.get("rating"),
                feedback_type="human",
                metadata={"state": state},
            )
            
            # Add feedback to the state
            if "metadata" not in state:
                state["metadata"] = {}
            
            if "feedback" not in state["metadata"]:
                state["metadata"]["feedback"] = {}
            
            state["metadata"]["feedback"]["human_feedback"] = feedback_entry
            
            return state
        
        return human_feedback_collector
    
    def create_conditional_improvement_edge(
        self,
        threshold: float = 7.0,
    ) -> Callable[[Dict[str, Any]], str]:
        """
        Create a conditional edge for a LangGraph workflow that routes to improvement
        if the evaluation score is below a threshold.
        
        Args:
            threshold: Score threshold for improvement.
            
        Returns:
            Conditional edge function.
        """
        def conditional_improvement_edge(state: Dict[str, Any]) -> str:
            """
            Conditional edge function.
            
            Args:
                state: Current state.
                
            Returns:
                Next node name.
            """
            # Check if feedback metadata exists
            if "metadata" in state and "feedback" in state["metadata"]:
                feedback = state["metadata"]["feedback"]
                
                # Check if evaluation score is below threshold
                if "evaluation_score" in feedback and feedback["evaluation_score"] < threshold:
                    return "improve"
            
            return "continue"
        
        return conditional_improvement_edge
    
    def analyze_feedback_history(
        self,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze feedback history to extract insights.
        
        Args:
            query: Filter by query.
            limit: Maximum number of feedback entries to analyze.
            
        Returns:
            Analysis results.
        """
        # Get feedback entries
        feedback_entries = self.feedback_collector.get_feedback(query=query)
        
        # Limit the number of entries
        feedback_entries = feedback_entries[:limit]
        
        # Analyze feedback
        analysis = self.feedback_collector.analyze_feedback(feedback_entries)
        
        # Calculate additional statistics
        human_feedback = [entry for entry in feedback_entries if entry["feedback_type"] == "human"]
        automated_feedback = [entry for entry in feedback_entries if entry["feedback_type"] == "automated"]
        
        human_ratings = [entry["rating"] for entry in human_feedback if entry["rating"] is not None]
        automated_ratings = [entry["rating"] for entry in automated_feedback if entry["rating"] is not None]
        
        human_avg_rating = sum(human_ratings) / len(human_ratings) if human_ratings else None
        automated_avg_rating = sum(automated_ratings) / len(automated_ratings) if automated_ratings else None
        
        # Add statistics to the analysis
        analysis["human_feedback_count"] = len(human_feedback)
        analysis["automated_feedback_count"] = len(automated_feedback)
        analysis["human_avg_rating"] = human_avg_rating
        analysis["automated_avg_rating"] = automated_avg_rating
        
        return analysis
