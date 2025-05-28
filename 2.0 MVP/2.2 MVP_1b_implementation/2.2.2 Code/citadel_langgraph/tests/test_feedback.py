
"""
Tests for feedback components.
"""

import os
import tempfile
import unittest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from citadel_langgraph.feedback import (
    BaseEvaluator,
    ResponseEvaluator,
    FeedbackCollector,
    SelfImprovementLoop,
    FeedbackOrchestrator,
)


class MockLLM:
    """Mock LLM for testing."""
    
    def invoke(self, prompt):
        """Mock invoke method."""
        if "evaluate" in prompt.lower():
            return """
            {
                "correctness_score": 0.8,
                "relevance_score": 0.7,
                "coherence_score": 0.9,
                "helpfulness_score": 0.6,
                "reasoning": "The response is mostly correct and coherent, but could be more helpful and relevant."
            }
            """
        elif "improve" in prompt.lower() or "refine" in prompt.lower():
            return "This is an improved response that addresses the feedback."
        elif "analyze" in prompt.lower():
            return """
            {
                "common_issues": ["Lack of detail", "Too technical"],
                "improvement_suggestions": ["Add more examples", "Simplify language"],
                "positive_patterns": ["Clear explanations", "Good structure"],
                "negative_patterns": ["Missing context", "Abrupt endings"]
            }
            """
        else:
            return "Mock LLM response"


class MockOllamaGateway:
    """Mock OllamaGateway for testing."""
    
    def get_llm(self, model=None):
        """Mock get_llm method."""
        return MockLLM()


class CustomEvaluator:
    """Custom evaluator for testing."""
    
    def invoke(self, inputs):
        """Mock invoke method."""
        return {
            "correctness_score": 0.8,
            "relevance_score": 0.7,
            "coherence_score": 0.9,
            "helpfulness_score": 0.6,
            "reasoning": "The response is mostly correct and coherent, but could be more helpful and relevant."
        }

class TestResponseEvaluator(unittest.TestCase):
    """Test cases for ResponseEvaluator."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a response evaluator with evaluator=None
        self.evaluator = ResponseEvaluator(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            evaluator=CustomEvaluator()  # Pass our custom evaluator
        )
    
    def test_evaluate(self):
        """Test evaluating a response."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        with patch.object(self.evaluator._evaluator, 'invoke', return_value={
            "correctness_score": 0.8,
            "relevance_score": 0.7,
            "coherence_score": 0.9,
            "helpfulness_score": 0.6,
            "reasoning": "The response is mostly correct and coherent, but could be more helpful and relevant."
        }):
            results = self.evaluator.evaluate(query, response)
        
        self.assertIn("overall_score", results)
        self.assertIn("passed", results)
        self.assertIn("criteria_scores", results)
        self.assertIn("feedback", results)
        
        # Check that scores are on a 1-10 scale
        self.assertGreaterEqual(results["overall_score"], 0)
        self.assertLessEqual(results["overall_score"], 10)
        
        for criterion, score in results["criteria_scores"].items():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 10)
    
    def test_get_improvement_suggestions(self):
        """Test getting improvement suggestions."""
        evaluation_results = {
            "overall_score": 7.5,
            "passed": True,
            "criteria_scores": {
                "correctness": 8.0,
                "relevance": 7.0,
                "coherence": 9.0,
                "helpfulness": 6.0,
            },
            "feedback": "The response is mostly correct and coherent, but could be more helpful and relevant."
        }
        
        suggestions = self.evaluator.get_improvement_suggestions(evaluation_results)
        self.assertIsInstance(suggestions, str)
        self.assertGreater(len(suggestions), 0)


class TestFeedbackCollector(unittest.TestCase):
    """Test cases for FeedbackCollector."""
    
    def setUp(self):
        """Set up test cases."""
        # Use a temporary directory for feedback storage
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_collector = FeedbackCollector(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            feedback_store_path=self.temp_dir,
        )
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_collect_feedback(self):
        """Test collecting feedback."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        feedback = "Good response, but could include more information about Paris."
        rating = 4
        
        feedback_entry = self.feedback_collector.collect_feedback(
            query=query,
            response=response,
            feedback=feedback,
            rating=rating,
        )
        
        self.assertIn("id", feedback_entry)
        self.assertIn("timestamp", feedback_entry)
        self.assertEqual(feedback_entry["query"], query)
        self.assertEqual(feedback_entry["response"], response)
        self.assertEqual(feedback_entry["feedback"], feedback)
        self.assertEqual(feedback_entry["rating"], rating)
    
    def test_get_feedback(self):
        """Test getting feedback."""
        # Add some feedback entries
        query = "What is the capital of France?"
        self.feedback_collector.collect_feedback(
            query=query,
            response="The capital of France is Paris.",
            feedback="Good response.",
            rating=4,
        )
        
        self.feedback_collector.collect_feedback(
            query=query,
            response="Paris is the capital of France.",
            feedback="Also good.",
            rating=5,
        )
        
        # Get feedback for the query
        feedback_entries = self.feedback_collector.get_feedback(query=query)
        
        self.assertEqual(len(feedback_entries), 2)
        self.assertEqual(feedback_entries[0]["query"], query)
        self.assertEqual(feedback_entries[1]["query"], query)
    
    def test_analyze_feedback(self):
        """Test analyzing feedback."""
        # Add some feedback entries
        query = "What is the capital of France?"
        self.feedback_collector.collect_feedback(
            query=query,
            response="The capital of France is Paris.",
            feedback="Good response.",
            rating=4,
        )
        
        self.feedback_collector.collect_feedback(
            query=query,
            response="Paris is the capital of France.",
            feedback="Also good.",
            rating=5,
        )
        
        # Analyze feedback
        analysis = self.feedback_collector.analyze_feedback(query=query)
        
        self.assertIn("count", analysis)
        self.assertIn("average_rating", analysis)
        self.assertEqual(analysis["count"], 2)
        self.assertEqual(analysis["average_rating"], 4.5)


class TestSelfImprovementLoop(unittest.TestCase):
    """Test cases for SelfImprovementLoop."""
    
    def setUp(self):
        """Set up test cases."""
        # Create the response evaluator with our custom evaluator
        self.evaluator = ResponseEvaluator(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            evaluator=CustomEvaluator()
        )
        
        self.improvement_loop = SelfImprovementLoop(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            evaluator=self.evaluator,
        )
    
    def test_refine_response(self):
        """Test refining a response."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        # Mock evaluation results
        evaluation_results = {
            "overall_score": 6.0,
            "passed": False,
            "criteria_scores": {
                "correctness": 8.0,
                "relevance": 5.0,
                "coherence": 7.0,
                "helpfulness": 4.0,
            },
            "feedback": "The response is correct but lacks detail and helpfulness."
        }
        
        # Mock evaluator methods
        with patch.object(self.evaluator, 'evaluate', return_value={
            "overall_score": 8.0,
            "passed": True,
            "criteria_scores": {
                "correctness": 8.0,
                "relevance": 8.0,
                "coherence": 8.0,
                "helpfulness": 8.0,
            },
            "feedback": "The response is now more detailed and helpful."
        }), patch.object(self.evaluator, 'get_improvement_suggestions', return_value="Add more details about Paris."):
            
            refinement_results = self.improvement_loop.refine_response(
                query=query,
                response=response,
                evaluation_results=evaluation_results,
            )
        
        self.assertIn("original_response", refinement_results)
        self.assertIn("refined_response", refinement_results)
        self.assertIn("original_score", refinement_results)
        self.assertIn("refined_score", refinement_results)
        self.assertIn("improvement", refinement_results)
        self.assertIn("iterations", refinement_results)
        
        self.assertEqual(refinement_results["original_response"], response)
        self.assertEqual(refinement_results["original_score"], 6.0)
        self.assertEqual(refinement_results["refined_score"], 8.0)
        self.assertEqual(refinement_results["improvement"], 2.0)
    
    def test_refine_prompt(self):
        """Test refining a prompt."""
        prompt = "Answer the following question about geography."
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        # Mock evaluation results
        evaluation_results = {
            "overall_score": 6.0,
            "passed": False,
            "criteria_scores": {
                "correctness": 8.0,
                "relevance": 5.0,
                "coherence": 7.0,
                "helpfulness": 4.0,
            },
            "feedback": "The response is correct but lacks detail and helpfulness."
        }
        
        refined_prompt = self.improvement_loop.refine_prompt(
            prompt=prompt,
            query=query,
            response=response,
            evaluation_results=evaluation_results,
        )
        
        self.assertIsInstance(refined_prompt, str)
        self.assertGreater(len(refined_prompt), 0)


class TestFeedbackOrchestrator(unittest.TestCase):
    """Test cases for FeedbackOrchestrator."""
    
    def setUp(self):
        """Set up test cases."""
        # Create the response evaluator with our custom evaluator
        self.evaluator = ResponseEvaluator(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            evaluator=CustomEvaluator()
        )
        
        self.feedback_collector = FeedbackCollector(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            feedback_store_path=tempfile.mkdtemp(),
        )
        
        self.improvement_loop = SelfImprovementLoop(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            evaluator=self.evaluator,
        )
        
        self.orchestrator = FeedbackOrchestrator(
            ollama_gateway=MockOllamaGateway(),
            model_name="mock-model",
            evaluator=self.evaluator,
            feedback_collector=self.feedback_collector,
            self_improvement_loop=self.improvement_loop,
        )
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.feedback_collector.feedback_store_path)
    
    def test_process_response(self):
        """Test processing a response."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        # Mock evaluator and improvement loop methods
        with patch.object(self.evaluator, 'evaluate', return_value={
            "overall_score": 6.0,
            "passed": False,
            "criteria_scores": {
                "correctness": 8.0,
                "relevance": 5.0,
                "coherence": 7.0,
                "helpfulness": 4.0,
            },
            "feedback": "The response is correct but lacks detail and helpfulness."
        }), patch.object(self.improvement_loop, 'refine_response', return_value={
            "original_response": response,
            "refined_response": "The capital of France is Paris, a city known for its culture and history.",
            "original_score": 6.0,
            "refined_score": 8.0,
            "improvement": 2.0,
            "iterations": 1,
            "evaluation_history": [],
        }):
            
            results = self.orchestrator.process_response(
                query=query,
                response=response,
            )
        
        self.assertIn("query", results)
        self.assertIn("original_response", results)
        self.assertIn("evaluation_results", results)
        self.assertIn("refinement_results", results)
        self.assertIn("final_response", results)
        
        self.assertEqual(results["query"], query)
        self.assertEqual(results["original_response"], response)
        self.assertNotEqual(results["final_response"], response)
    
    def test_create_feedback_node(self):
        """Test creating a feedback node."""
        # Define extractors and updaters
        def query_extractor(state):
            return state["query"]
        
        def response_extractor(state):
            return state["response"]
        
        def response_updater(state, response):
            state["response"] = response
            return state
        
        # Create feedback node
        feedback_node = self.orchestrator.create_feedback_node(
            query_extractor=query_extractor,
            response_extractor=response_extractor,
            response_updater=response_updater,
        )
        
        # Mock evaluator and improvement loop methods
        with patch.object(self.evaluator, 'evaluate', return_value={
            "overall_score": 6.0,
            "passed": False,
            "criteria_scores": {
                "correctness": 8.0,
                "relevance": 5.0,
                "coherence": 7.0,
                "helpfulness": 4.0,
            },
            "feedback": "The response is correct but lacks detail and helpfulness."
        }), patch.object(self.improvement_loop, 'refine_response', return_value={
            "original_response": "The capital of France is Paris.",
            "refined_response": "The capital of France is Paris, a city known for its culture and history.",
            "original_score": 6.0,
            "refined_score": 8.0,
            "improvement": 2.0,
            "iterations": 1,
            "evaluation_history": [],
        }):
            
            # Test the feedback node
            state = {
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
            }
            
            updated_state = feedback_node(state)
        
        self.assertIn("metadata", updated_state)
        self.assertIn("feedback", updated_state["metadata"])
        self.assertIn("evaluation_score", updated_state["metadata"]["feedback"])
        self.assertIn("passed_evaluation", updated_state["metadata"]["feedback"])
        self.assertIn("was_refined", updated_state["metadata"]["feedback"])
        
        self.assertEqual(updated_state["response"], "The capital of France is Paris, a city known for its culture and history.")


if __name__ == "__main__":
    unittest.main()
