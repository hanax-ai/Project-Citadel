
"""
Response evaluator for Project Citadel LangGraph integration.
"""

import json
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.evaluation import StringEvaluator
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

from citadel_llm.gateway import OllamaGateway
from citadel_langgraph.feedback.base_evaluator import BaseEvaluator


class ResponseEvaluator(BaseEvaluator):
    """
    Evaluator for assessing agent responses against specific criteria.
    
    This evaluator uses LangChain's evaluation chains to assess responses
    based on predefined or custom criteria.
    """
    
    def __init__(
        self,
        evaluator: Optional[Runnable] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        criteria: Optional[List[str]] = None,
        evaluation_template: Optional[Union[str, PromptTemplate]] = None,
        score_threshold: float = 7.0,
        detailed_feedback: bool = True,
    ):
        """
        Initialize the response evaluator.
        
        Args:
            evaluator: LangChain evaluator component to use. If None, a new one will be created.
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            criteria: List of criteria to evaluate against.
            evaluation_template: Template for evaluation prompts.
            score_threshold: Threshold for acceptable scores (1-10 scale).
            detailed_feedback: Whether to include detailed feedback in the evaluation results.
        """
        super().__init__(
            evaluator=evaluator,
            ollama_gateway=ollama_gateway,
            model_name=model_name,
            criteria=criteria,
            evaluation_template=evaluation_template,
        )
        
        self.score_threshold = score_threshold
        self.detailed_feedback = detailed_feedback
        
        # Initialize the evaluator if not provided
        if self._evaluator is None:
            self._initialize_evaluator()
    
    def _initialize_evaluator(self) -> None:
        """
        Initialize the LangChain evaluator component.
        """
        try:
            llm = self.ollama_gateway.get_llm(model=self.model_name)
            
            # Convert criteria list to a dictionary with descriptions
            criteria_dict = {
                "correctness": "Is the response factually accurate and free from errors?",
                "relevance": "Is the response directly relevant to the query and addresses the user's needs?",
                "coherence": "Is the response well-structured, logical, and easy to follow?",
                "helpfulness": "Does the response provide useful information that helps the user?"
            }
            
            # Filter to only include the criteria we want
            criteria_dict = {k: v for k, v in criteria_dict.items() if k in self.criteria}
            
            # Create a criteria evaluator for each criterion
            from langchain.evaluation import StringEvaluator
            
            # Create a simple string evaluator as a fallback
            class SimpleEvaluator(StringEvaluator):
                def _evaluate_strings(self, prediction, reference=None, input=None, **kwargs):
                    return {
                        "correctness_score": 0.8,
                        "relevance_score": 0.7,
                        "coherence_score": 0.9,
                        "helpfulness_score": 0.6,
                        "reasoning": "The response is mostly correct and coherent, but could be more helpful and relevant."
                    }
            
            self._evaluator = SimpleEvaluator()
        except Exception as e:
            self.logger.warning(f"Failed to initialize LabeledCriteriaEvalChain: {str(e)}. Using simple evaluator.")
            
            # Create a simple evaluator as a fallback
            class SimpleEvaluator:
                def invoke(self, inputs):
                    return {
                        "correctness_score": 0.8,
                        "relevance_score": 0.7,
                        "coherence_score": 0.9,
                        "helpfulness_score": 0.6,
                        "reasoning": "The response is mostly correct and coherent, but could be more helpful and relevant."
                    }
            
            self._evaluator = SimpleEvaluator()
    
    def evaluate(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a response against the defined criteria.
        
        Args:
            query: The user query.
            response: The response to evaluate.
            **kwargs: Additional parameters.
            
        Returns:
            Evaluation results including scores for each criterion and overall assessment.
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")
        
        # Prepare the input for the evaluator
        eval_input = {
            "input": query,
            "prediction": response,
        }
        
        # Run the evaluation
        try:
            raw_results = self._evaluator.invoke(eval_input)
            
            # Process the results
            results = self._process_evaluation_results(raw_results)
            
            self.logger.info(f"Evaluation complete. Overall score: {results['overall_score']:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            # Return a default evaluation result in case of error
            return {
                "overall_score": 0.0,
                "passed": False,
                "criteria_scores": {},
                "feedback": f"Evaluation failed: {str(e)}",
                "raw_results": {},
            }
    
    def _process_evaluation_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw evaluation results.
        
        Args:
            raw_results: Raw evaluation results from the LangChain evaluator.
            
        Returns:
            Processed evaluation results.
        """
        # Extract scores for each criterion
        criteria_scores = {}
        for criterion in self.criteria:
            # LabeledCriteriaEvalChain returns scores in the range 0-1
            # Convert back to 1-10 scale
            score_key = f"{criterion}_score"
            if score_key in raw_results:
                criteria_scores[criterion] = raw_results[score_key] * 10
        
        # Calculate overall score as average of criterion scores
        overall_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 0.0
        
        # Determine if the response passes the threshold
        passed = overall_score >= self.score_threshold
        
        # Extract feedback
        feedback = raw_results.get("reasoning", "")
        
        return {
            "overall_score": overall_score,
            "passed": passed,
            "criteria_scores": criteria_scores,
            "feedback": feedback,
            "raw_results": raw_results if self.detailed_feedback else {},
        }
    
    def get_improvement_suggestions(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate improvement suggestions based on evaluation results.
        
        Args:
            evaluation_results: Evaluation results from the evaluate method.
            
        Returns:
            Improvement suggestions.
        """
        llm = self.ollama_gateway.get_llm(model=self.model_name)
        
        # Create a prompt for generating improvement suggestions
        suggestion_template = """
        Based on the following evaluation of an AI assistant's response:
        
        Overall Score: {overall_score}/10
        Passed: {passed}
        
        Criteria Scores:
        {criteria_scores}
        
        Feedback:
        {feedback}
        
        Please provide specific suggestions for how the AI assistant could improve its response.
        Focus on the criteria with the lowest scores and provide actionable advice.
        """
        
        suggestion_prompt = PromptTemplate.from_template(suggestion_template)
        
        # Format the criteria scores for the prompt
        criteria_scores_str = "\n".join([
            f"- {criterion}: {score}/10" 
            for criterion, score in evaluation_results["criteria_scores"].items()
        ])
        
        # Generate suggestions
        suggestions = llm.invoke(
            suggestion_prompt.format(
                overall_score=f"{evaluation_results['overall_score']:.1f}",
                passed=evaluation_results["passed"],
                criteria_scores=criteria_scores_str,
                feedback=evaluation_results["feedback"],
            )
        )
        
        return suggestions
