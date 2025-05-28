
"""
Agent workflows for LangGraph.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union

from langgraph.graph import StateGraph, END

from citadel_core.logging import get_logger
from citadel_llm.gateway import OllamaGateway
from citadel_llm.models import LLMManager

from citadel_langgraph.state.agent_state import (
    AgentState,
    ReActAgentState,
    MultiAgentState,
    create_agent_state,
    create_react_agent_state,
    create_multi_agent_state,
)
from citadel_langgraph.nodes.agent_nodes import (
    LLMNode,
    ReActAgentNode,
    ToolExecutionNode,
    AgentCoordinatorNode,
)
from citadel_langgraph.edges.agent_edges import (
    ReActAgentStepEdge,
    ActionTypeEdge,
    FinalAnswerEdge,
    MultiAgentCoordinationEdge,
    TaskCompletionEdge,
)
from citadel_langgraph.feedback import (
    ResponseEvaluator,
    FeedbackCollector,
    SelfImprovementLoop,
    FeedbackOrchestrator,
)
from .base import BaseWorkflow


class LLMAgentWorkflow(BaseWorkflow[AgentState]):
    """
    Workflow for a simple LLM agent.
    
    This workflow handles basic LLM interactions.
    """
    
    def __init__(
        self,
        name: str = "llm_agent",
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        system_message: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the LLM agent workflow.
        
        Args:
            name: Name of the workflow.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            system_message: System message to use.
            logger: Logger instance.
        """
        super().__init__(name, AgentState, logger)
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.system_message = system_message
    
    def build(self) -> None:
        """
        Build the LLM agent workflow.
        """
        self.logger.info(f"Building LLM agent workflow '{self.name}'")
        
        # Create nodes
        llm_node = LLMNode(
            name="generate_response",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            system_message=self.system_message,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(llm_node)
        
        # Set entry point
        self.set_entry_point("generate_response")
        
        # Add end node
        self.add_end_node("generate_response")
    
    def run(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """
        Run the LLM agent workflow.
        
        Args:
            messages: Optional list of messages.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_agent_state(
            system_message=self.system_message,
            metadata=metadata,
        )
        
        # Add messages to state
        if messages:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            parsed_messages = []
            for msg in messages:
                if msg.get("role") == "user":
                    parsed_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    parsed_messages.append(AIMessage(content=msg.get("content", "")))
                elif msg.get("role") == "system":
                    parsed_messages.append(SystemMessage(content=msg.get("content", "")))
            
            initial_state["messages"] = parsed_messages
        
        # Run the workflow
        return super().run(initial_state)


class ReActAgentWorkflow(BaseWorkflow[ReActAgentState]):
    """
    Workflow for a ReAct agent.
    
    This workflow implements the ReAct (Reasoning and Acting) pattern.
    """
    
    def __init__(
        self,
        name: str = "react_agent",
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        system_message: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ReAct agent workflow.
        
        Args:
            name: Name of the workflow.
            llm_manager: LLM manager to use.
            ollama_gateway: OllamaGateway instance to use.
            model_name: Name of the model to use.
            system_message: System message to use.
            tools: Dictionary mapping tool names to functions.
            logger: Logger instance.
        """
        super().__init__(name, ReActAgentState, logger)
        
        # Use provided LLM manager or create a new one
        self.llm_manager = llm_manager or LLMManager(
            gateway=ollama_gateway or OllamaGateway()
        )
        self.model_name = model_name
        self.system_message = system_message
        self.tools = tools or {}
    
    def build(self) -> None:
        """
        Build the ReAct agent workflow.
        """
        self.logger.info(f"Building ReAct agent workflow '{self.name}'")
        
        # Convert tools to the format expected by ReActAgentNode
        tool_descriptions = [
            {
                "name": name,
                "description": func.__doc__ or f"Tool for {name}",
            }
            for name, func in self.tools.items()
        ]
        
        # Create nodes
        react_node = ReActAgentNode(
            name="react",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            system_message=self.system_message,
            tools=tool_descriptions,
            logger=self.logger,
        )
        
        tool_execution_node = ToolExecutionNode(
            name="execute_tool",
            tool_registry=self.tools,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(react_node)
        self.add_node(tool_execution_node)
        
        # Add edges
        react_step_edge = ReActAgentStepEdge(
            name="react_step",
            step_routes={
                "execute_action": "execute_tool",
                "react": "react",
                "finish": END,
            },
            default_route="react",
        )
        
        self.add_edge("react", react_step_edge)
        self.add_edge("execute_tool", "react")
        
        # Set entry point
        self.set_entry_point("react")
    
    def run(
        self,
        input_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReActAgentState:
        """
        Run the ReAct agent workflow.
        
        Args:
            input_message: Input message to the agent.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_react_agent_state(
            system_message=self.system_message,
            tools=[
                {
                    "name": name,
                    "description": func.__doc__ or f"Tool for {name}",
                }
                for name, func in self.tools.items()
            ],
            metadata=metadata,
        )
        
        # Add input message to state
        from langchain_core.messages import HumanMessage
        initial_state["messages"] = [HumanMessage(content=input_message)]
        
        # Run the workflow
        return super().run(initial_state)


class FeedbackEnabledAgentWorkflow(BaseWorkflow[AgentState]):
    """
    Workflow for an LLM agent with feedback loops.
    
    This workflow incorporates evaluation, feedback collection, and self-improvement
    to enhance the quality of agent responses.
    """
    
    def __init__(
        self,
        name: str = "feedback_enabled_agent",
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        system_message: str = "You are a helpful AI assistant.",
        evaluator: Optional[ResponseEvaluator] = None,
        feedback_collector: Optional[FeedbackCollector] = None,
        self_improvement_loop: Optional[SelfImprovementLoop] = None,
        feedback_orchestrator: Optional[FeedbackOrchestrator] = None,
        auto_improve: bool = True,
        evaluation_threshold: float = 7.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the feedback-enabled agent workflow.
        
        Args:
            name: Name of the workflow.
            llm_manager: LLM manager instance.
            ollama_gateway: OllamaGateway instance.
            model_name: Name of the model to use.
            system_message: System message for the agent.
            evaluator: ResponseEvaluator instance.
            feedback_collector: FeedbackCollector instance.
            self_improvement_loop: SelfImprovementLoop instance.
            feedback_orchestrator: FeedbackOrchestrator instance.
            auto_improve: Whether to automatically improve responses.
            evaluation_threshold: Threshold for acceptable evaluation scores.
            logger: Logger instance.
        """
        super().__init__(name, AgentState, logger)
        
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.llm_manager = llm_manager or LLMManager()
        self.model_name = model_name
        self.system_message = system_message
        self.auto_improve = auto_improve
        self.evaluation_threshold = evaluation_threshold
        
        # Initialize feedback components
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
        
        self.feedback_orchestrator = feedback_orchestrator or FeedbackOrchestrator(
            ollama_gateway=self.ollama_gateway,
            model_name=self.model_name,
            evaluator=self.evaluator,
            feedback_collector=self.feedback_collector,
            self_improvement_loop=self.self_improvement_loop,
            auto_improve=self.auto_improve,
        )
        
        # Build the workflow
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """
        Build the workflow graph.
        """
        # Create nodes
        llm_node = LLMNode(
            name="llm",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            system_message=self.system_message,
            logger=self.logger,
        )
        
        # Create a feedback node
        def feedback_node(state: AgentState) -> AgentState:
            """
            Process the agent's response through the feedback pipeline.
            """
            # Extract query and response
            query = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
            response = state["messages"][-1].content if state["messages"] else ""
            
            # Process the response
            results = self.feedback_orchestrator.process_response(query, response)
            
            # Update the state with the processed response
            if results["final_response"] != response:
                # Replace the last message with the improved response
                from langchain_core.messages import AIMessage
                state["messages"][-1] = AIMessage(content=results["final_response"])
            
            # Add feedback metadata to the state
            if "metadata" not in state:
                state["metadata"] = {}
            
            state["metadata"]["feedback"] = {
                "evaluation_score": results["evaluation_results"]["overall_score"],
                "passed_evaluation": results["evaluation_results"]["passed"],
                "was_refined": results["refinement_results"] is not None,
            }
            
            if results["refinement_results"]:
                state["metadata"]["feedback"]["refinement"] = {
                    "original_score": results["refinement_results"]["original_score"],
                    "refined_score": results["refinement_results"]["refined_score"],
                    "improvement": results["refinement_results"]["improvement"],
                    "iterations": results["refinement_results"]["iterations"],
                }
            
            return state
        
        # Create a conditional edge for feedback
        def feedback_edge(state: AgentState) -> str:
            """
            Determine whether to apply feedback.
            """
            # Check if this is the final response
            if state.get("final_answer", False):
                # Apply feedback to final answers
                return "feedback"
            
            # Skip feedback for intermediate steps
            return END
        
        # Add nodes to the workflow
        self.add_node(llm_node)
        self.add_node(feedback_node)
        
        # Add edges
        self.add_edge("llm", feedback_edge)
        self.add_edge("feedback", END)
        
        # Set entry point
        self.set_entry_point("llm")
    
    def run(
        self,
        input_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        collect_human_feedback: bool = False,
    ) -> AgentState:
        """
        Run the feedback-enabled agent workflow.
        
        Args:
            input_message: Input message to the agent.
            metadata: Optional workflow metadata.
            collect_human_feedback: Whether to collect human feedback after the run.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_agent_state(
            system_message=self.system_message,
            metadata=metadata,
        )
        
        # Add input message to state
        from langchain_core.messages import HumanMessage
        initial_state["messages"] = [HumanMessage(content=input_message)]
        
        # Run the workflow
        final_state = super().run(initial_state)
        
        # Collect human feedback if requested
        if collect_human_feedback and final_state["messages"]:
            query = input_message
            response = final_state["messages"][-1].content if final_state["messages"] else ""
            
            # This would typically be integrated with a UI to collect feedback
            self.logger.info("Human feedback collection would be triggered here.")
            self.logger.info(f"Query: {query}")
            self.logger.info(f"Response: {response}")
        
        return final_state


class FeedbackEnabledReActAgentWorkflow(BaseWorkflow[ReActAgentState]):
    """
    Workflow for a ReAct agent with feedback loops.
    
    This workflow incorporates evaluation, feedback collection, and self-improvement
    to enhance the quality of agent responses in a ReAct framework.
    """
    
    def __init__(
        self,
        name: str = "feedback_enabled_react_agent",
        llm_manager: Optional[LLMManager] = None,
        ollama_gateway: Optional[OllamaGateway] = None,
        model_name: str = "mistral:latest",
        system_message: str = "You are a helpful AI assistant that can use tools.",
        tools: Dict[str, Callable] = None,
        evaluator: Optional[ResponseEvaluator] = None,
        feedback_collector: Optional[FeedbackCollector] = None,
        self_improvement_loop: Optional[SelfImprovementLoop] = None,
        feedback_orchestrator: Optional[FeedbackOrchestrator] = None,
        auto_improve: bool = True,
        evaluation_threshold: float = 7.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the feedback-enabled ReAct agent workflow.
        
        Args:
            name: Name of the workflow.
            llm_manager: LLM manager instance.
            ollama_gateway: OllamaGateway instance.
            model_name: Name of the model to use.
            system_message: System message for the agent.
            tools: Dictionary of tools available to the agent.
            evaluator: ResponseEvaluator instance.
            feedback_collector: FeedbackCollector instance.
            self_improvement_loop: SelfImprovementLoop instance.
            feedback_orchestrator: FeedbackOrchestrator instance.
            auto_improve: Whether to automatically improve responses.
            evaluation_threshold: Threshold for acceptable evaluation scores.
            logger: Logger instance.
        """
        super().__init__(name, ReActAgentState, logger)
        
        self.ollama_gateway = ollama_gateway or OllamaGateway()
        self.llm_manager = llm_manager or LLMManager()
        self.model_name = model_name
        self.system_message = system_message
        self.tools = tools or {}
        self.auto_improve = auto_improve
        self.evaluation_threshold = evaluation_threshold
        
        # Initialize feedback components
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
        
        self.feedback_orchestrator = feedback_orchestrator or FeedbackOrchestrator(
            ollama_gateway=self.ollama_gateway,
            model_name=self.model_name,
            evaluator=self.evaluator,
            feedback_collector=self.feedback_collector,
            self_improvement_loop=self.self_improvement_loop,
            auto_improve=self.auto_improve,
        )
        
        # Build the workflow
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """
        Build the workflow graph.
        """
        # Create tool descriptions
        tool_descriptions = [
            {
                "name": name,
                "description": func.__doc__ or f"Tool for {name}",
            }
            for name, func in self.tools.items()
        ]
        
        # Create nodes
        react_node = ReActAgentNode(
            name="react",
            llm_manager=self.llm_manager,
            model_name=self.model_name,
            system_message=self.system_message,
            tools=tool_descriptions,
            logger=self.logger,
        )
        
        tool_execution_node = ToolExecutionNode(
            name="execute_tool",
            tool_registry=self.tools,
            logger=self.logger,
        )
        
        # Create a feedback node
        def feedback_node(state: ReActAgentState) -> ReActAgentState:
            """
            Process the agent's final response through the feedback pipeline.
            """
            # Extract query and response
            query = state["messages"][0].content if state["messages"] else ""
            response = state["final_answer"] if "final_answer" in state else ""
            
            if not response:
                return state
            
            # Process the response
            results = self.feedback_orchestrator.process_response(query, response)
            
            # Update the state with the processed response
            if results["final_response"] != response:
                # Replace the final answer with the improved response
                state["final_answer"] = results["final_response"]
            
            # Add feedback metadata to the state
            if "metadata" not in state:
                state["metadata"] = {}
            
            state["metadata"]["feedback"] = {
                "evaluation_score": results["evaluation_results"]["overall_score"],
                "passed_evaluation": results["evaluation_results"]["passed"],
                "was_refined": results["refinement_results"] is not None,
            }
            
            if results["refinement_results"]:
                state["metadata"]["feedback"]["refinement"] = {
                    "original_score": results["refinement_results"]["original_score"],
                    "refined_score": results["refinement_results"]["refined_score"],
                    "improvement": results["refinement_results"]["improvement"],
                    "iterations": results["refinement_results"]["iterations"],
                }
            
            return state
        
        # Add nodes to the workflow
        self.add_node(react_node)
        self.add_node(tool_execution_node)
        self.add_node(feedback_node)
        
        # Add edges
        react_step_edge = ReActAgentStepEdge(
            name="react_step",
            step_routes={
                "execute_action": "execute_tool",
                "react": "react",
                "finish": "feedback",  # Route to feedback node before ending
            },
            default_route="react",
        )
        
        self.add_edge("react", react_step_edge)
        self.add_edge("execute_tool", "react")
        self.add_edge("feedback", END)
        
        # Set entry point
        self.set_entry_point("react")
    
    def run(
        self,
        input_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        collect_human_feedback: bool = False,
    ) -> ReActAgentState:
        """
        Run the feedback-enabled ReAct agent workflow.
        
        Args:
            input_message: Input message to the agent.
            metadata: Optional workflow metadata.
            collect_human_feedback: Whether to collect human feedback after the run.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_react_agent_state(
            system_message=self.system_message,
            tools=[
                {
                    "name": name,
                    "description": func.__doc__ or f"Tool for {name}",
                }
                for name, func in self.tools.items()
            ],
            metadata=metadata,
        )
        
        # Add input message to state
        from langchain_core.messages import HumanMessage
        initial_state["messages"] = [HumanMessage(content=input_message)]
        
        # Run the workflow
        final_state = super().run(initial_state)
        
        # Collect human feedback if requested
        if collect_human_feedback and "final_answer" in final_state:
            query = input_message
            response = final_state["final_answer"]
            
            # This would typically be integrated with a UI to collect feedback
            self.logger.info("Human feedback collection would be triggered here.")
            self.logger.info(f"Query: {query}")
            self.logger.info(f"Response: {response}")
        
        return final_state


class MultiAgentWorkflow(BaseWorkflow[MultiAgentState]):
    """
    Workflow for multiple agents.
    
    This workflow coordinates multiple agents to solve a task.
    """
    
    def __init__(
        self,
        name: str = "multi_agent",
        agent_configs: Dict[str, Dict[str, Any]] = None,
        agent_workflows: Dict[str, BaseWorkflow] = None,
        agent_selection_strategy: Optional[Callable[[MultiAgentState], str]] = None,
        task_completion_condition: Optional[Callable[[MultiAgentState], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the multi-agent workflow.
        
        Args:
            name: Name of the workflow.
            agent_configs: Dictionary mapping agent IDs to their configurations.
            agent_workflows: Dictionary mapping agent IDs to their workflows.
            agent_selection_strategy: Function to select the next agent.
            task_completion_condition: Function to determine if the task is complete.
            logger: Logger instance.
        """
        super().__init__(name, MultiAgentState, logger)
        
        self.agent_configs = agent_configs or {}
        self.agent_workflows = agent_workflows or {}
        
        # Default agent selection strategy: round-robin
        self.agent_selection_strategy = agent_selection_strategy or self._default_agent_selection
        
        # Default task completion condition: check if status is "completed"
        self.task_completion_condition = task_completion_condition or self._default_task_completion
    
    def _default_agent_selection(self, state: MultiAgentState) -> str:
        """
        Default agent selection strategy: round-robin.
        
        Args:
            state: Current workflow state.
            
        Returns:
            ID of the next agent.
        """
        agent_ids = list(state.get("agent_states", {}).keys())
        if not agent_ids:
            raise ValueError("No agents defined")
        
        active_agent = state.get("active_agent")
        if not active_agent or active_agent not in agent_ids:
            return agent_ids[0]
        
        # Get the next agent in the list
        current_index = agent_ids.index(active_agent)
        next_index = (current_index + 1) % len(agent_ids)
        return agent_ids[next_index]
    
    def _default_task_completion(self, state: MultiAgentState) -> bool:
        """
        Default task completion condition: check if status is "completed".
        
        Args:
            state: Current workflow state.
            
        Returns:
            True if the task is complete, False otherwise.
        """
        return state.get("status") == "completed"
    
    def build(self) -> None:
        """
        Build the multi-agent workflow.
        """
        self.logger.info(f"Building multi-agent workflow '{self.name}'")
        
        # Create nodes
        coordinator_node = AgentCoordinatorNode(
            name="coordinate_agents",
            agent_selection_strategy=self.agent_selection_strategy,
            logger=self.logger,
        )
        
        # Add nodes to the workflow
        self.add_node(coordinator_node)
        
        # Add agent nodes
        for agent_id, workflow in self.agent_workflows.items():
            # Compile the agent workflow
            compiled_workflow = workflow.compile()
            
            # Add the agent node
            self.add_node(compiled_workflow)
        
        # Add edges
        agent_coordination_edge = MultiAgentCoordinationEdge(
            name="agent_coordination",
            agent_routes={
                agent_id: agent_id
                for agent_id in self.agent_workflows
            },
            default_route="coordinate_agents",
        )
        
        task_completion_edge = TaskCompletionEdge(
            name="task_completion",
            completion_condition=self.task_completion_condition,
            complete_node=END,
            incomplete_node="coordinate_agents",
        )
        
        self.add_edge("coordinate_agents", agent_coordination_edge)
        
        # Add edges from agent nodes back to coordinator
        for agent_id in self.agent_workflows:
            self.add_edge(agent_id, task_completion_edge)
        
        # Set entry point
        self.set_entry_point("coordinate_agents")
    
    def run(
        self,
        input_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultiAgentState:
        """
        Run the multi-agent workflow.
        
        Args:
            input_message: Input message to the agents.
            metadata: Optional workflow metadata.
            
        Returns:
            Final state.
        """
        # Create initial state
        initial_state = create_multi_agent_state(
            agent_configs=self.agent_configs,
            metadata=metadata,
        )
        
        # Add input message to shared memory
        initial_state["shared_memory"] = {
            "input_message": input_message,
        }
        
        # Run the workflow
        return super().run(initial_state)
