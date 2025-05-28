
"""
Feedback collector for Project Citadel LangGraph integration.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway


class FeedbackCollector:
    """
    Component for collecting and processing human feedback.
    
    This class provides methods for collecting, storing, and analyzing
    human feedback on agent responses.
    """
    
    def __init__(
        self,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        feedback_store_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the feedback collector.
        
        Args:
            ollama_gateway: OllamaGateway instance to use. If None, a new one will be created.
            model_name: Name of the model to use.
            feedback_store_path: Path to store feedback data. If None, a default path will be used.
            logger: Logger instance.
        """
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.model_name = model_name
        self.feedback_store_path = feedback_store_path or os.path.join(
            os.path.expanduser("~"), ".citadel", "feedback"
        )
        self.logger = logger or get_logger("citadel.langgraph.feedback.feedback_collector")
        
        # Create the feedback store directory if it doesn't exist
        os.makedirs(self.feedback_store_path, exist_ok=True)
        
        # Initialize the feedback store
        self._feedback_store = self._load_feedback_store()
    
    def _load_feedback_store(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the feedback store from disk.
        
        Returns:
            Feedback store.
        """
        store_file = os.path.join(self.feedback_store_path, "feedback_store.json")
        
        if os.path.exists(store_file):
            try:
                with open(store_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading feedback store: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_feedback_store(self) -> None:
        """
        Save the feedback store to disk.
        """
        store_file = os.path.join(self.feedback_store_path, "feedback_store.json")
        
        try:
            with open(store_file, "w") as f:
                json.dump(self._feedback_store, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving feedback store: {str(e)}")
    
    def collect_feedback(
        self,
        query: str,
        response: str,
        feedback: str,
        rating: Optional[int] = None,
        feedback_type: str = "human",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Collect feedback on a response.
        
        Args:
            query: The user query.
            response: The response to collect feedback on.
            feedback: The feedback text.
            rating: Optional numerical rating (e.g., 1-5).
            feedback_type: Type of feedback (e.g., "human", "automated").
            metadata: Additional metadata.
            
        Returns:
            The collected feedback entry.
        """
        self.logger.info(f"Collecting {feedback_type} feedback for query: {query[:50]}...")
        
        # Create a feedback entry
        feedback_entry = {
            "id": f"feedback_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(query + response) % 10000}",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "feedback": feedback,
            "rating": rating,
            "feedback_type": feedback_type,
            "metadata": metadata or {},
        }
        
        # Add the feedback to the store
        query_hash = str(hash(query) % 100000)
        if query_hash not in self._feedback_store:
            self._feedback_store[query_hash] = []
        
        self._feedback_store[query_hash].append(feedback_entry)
        
        # Save the feedback store
        self._save_feedback_store()
        
        self.logger.info(f"Feedback collected and stored with ID: {feedback_entry['id']}")
        
        return feedback_entry
    
    def get_feedback(
        self,
        query: Optional[str] = None,
        feedback_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get feedback entries.
        
        Args:
            query: Filter by query.
            feedback_id: Filter by feedback ID.
            
        Returns:
            List of feedback entries.
        """
        if feedback_id:
            # Search for the feedback entry by ID
            for entries in self._feedback_store.values():
                for entry in entries:
                    if entry["id"] == feedback_id:
                        return [entry]
            return []
        
        if query:
            # Search for feedback entries by query
            query_hash = str(hash(query) % 100000)
            return self._feedback_store.get(query_hash, [])
        
        # Return all feedback entries
        all_entries = []
        for entries in self._feedback_store.values():
            all_entries.extend(entries)
        
        return all_entries
    
    def analyze_feedback(
        self,
        feedback_entries: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze feedback entries to extract insights.
        
        Args:
            feedback_entries: Feedback entries to analyze. If None, all entries will be used.
            query: Filter by query.
            
        Returns:
            Analysis results.
        """
        # Get the feedback entries to analyze
        if feedback_entries is None:
            feedback_entries = self.get_feedback(query=query)
        
        if not feedback_entries:
            return {
                "count": 0,
                "average_rating": None,
                "common_issues": [],
                "improvement_suggestions": [],
            }
        
        # Calculate statistics
        ratings = [entry["rating"] for entry in feedback_entries if entry["rating"] is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Use LLM to analyze feedback and extract common issues and improvement suggestions
        llm = self.ollama_gateway.get_llm(model=self.model_name)
        
        analysis_template = """
        Please analyze the following user feedback on AI assistant responses:
        
        {feedback_entries}
        
        Identify:
        1. Common issues or complaints mentioned in the feedback
        2. Specific suggestions for improvement
        3. Patterns in what users liked or disliked
        
        Provide your analysis in JSON format with the following structure:
        ```json
        {
            "common_issues": ["issue1", "issue2", ...],
            "improvement_suggestions": ["suggestion1", "suggestion2", ...],
            "positive_patterns": ["pattern1", "pattern2", ...],
            "negative_patterns": ["pattern1", "pattern2", ...]
        }
        ```
        """
        
        analysis_prompt = PromptTemplate.from_template(analysis_template)
        
        # Format the feedback entries for the prompt
        feedback_entries_str = "\n\n".join([
            f"Query: {entry['query']}\nResponse: {entry['response']}\nFeedback: {entry['feedback']}" +
            (f"\nRating: {entry['rating']}" if entry['rating'] is not None else "")
            for entry in feedback_entries[:10]  # Limit to 10 entries to avoid token limits
        ])
        
        # Generate analysis
        try:
            analysis_text = llm.invoke(
                analysis_prompt.format(
                    feedback_entries=feedback_entries_str,
                )
            )
            
            # Extract JSON from the response
            analysis_json = self._extract_json(analysis_text)
            
            return {
                "count": len(feedback_entries),
                "average_rating": average_rating,
                "common_issues": analysis_json.get("common_issues", []),
                "improvement_suggestions": analysis_json.get("improvement_suggestions", []),
                "positive_patterns": analysis_json.get("positive_patterns", []),
                "negative_patterns": analysis_json.get("negative_patterns", []),
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback: {str(e)}")
            return {
                "count": len(feedback_entries),
                "average_rating": average_rating,
                "common_issues": [],
                "improvement_suggestions": [],
                "error": str(e),
            }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON.
            
        Returns:
            Extracted JSON.
        """
        try:
            # Try to parse the entire text as JSON
            return json.loads(text)
        except:
            # Try to extract JSON from the text
            start = text.find("{")
            end = text.rfind("}")
            
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end+1])
                except:
                    pass
            
            # Try to extract JSON from code blocks
            import re
            json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
            
            for block in json_blocks:
                try:
                    return json.loads(block)
                except:
                    continue
            
            # Return empty dict if no JSON found
            return {}
    
    def clear_feedback(self, query: Optional[str] = None) -> None:
        """
        Clear feedback entries.
        
        Args:
            query: Filter by query. If None, all entries will be cleared.
        """
        if query:
            # Clear feedback entries for a specific query
            query_hash = str(hash(query) % 100000)
            if query_hash in self._feedback_store:
                del self._feedback_store[query_hash]
        else:
            # Clear all feedback entries
            self._feedback_store = {}
        
        # Save the feedback store
        self._save_feedback_store()
        
        self.logger.info(f"Feedback cleared for query: {query if query else 'all'}")
