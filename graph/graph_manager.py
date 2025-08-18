"""
LangGraph workflow orchestration manager.

This module provides the main orchestration for LangGraph workflows,
managing state transitions and coordinating between different analysis nodes.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, TypedDict
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.nodes.content_analyzer import ContentAnalyzerNode, ContentAnalysisState, ContentType
from graph.nodes.risk_assessor import RiskAssessorNode
from graph.nodes.report_generator import ReportGeneratorNode
from graph.nodes.source_verifier import SourceVerifierNode
from llm.langchain_pipeline import LangChainPipeline
from llm.llm_router import LLMRouter
from utils.logger import get_logger
from utils.exceptions import ContentProcessingError

logger = get_logger(__name__)


class WorkflowType(Enum):
    """Available workflow types."""
    TEXT_ANALYSIS = "text_analysis"
    IMAGE_ANALYSIS = "image_analysis"
    URL_ANALYSIS = "url_analysis"


class GraphState(TypedDict):
    """State schema for LangGraph workflows."""
    content_type: str
    content: str
    metadata: Dict[str, Any]
    analysis_results: Dict[str, Any]
    error: Optional[str]
    processing_stage: str
    workflow_config: Dict[str, Any]


class GraphManager:
    """
    LangGraph workflow orchestration manager.
    
    Manages the creation and execution of different analysis workflows
    with proper state management and error recovery.
    """
    
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        self.pipeline = LangChainPipeline(llm_router)
        self.logger = get_logger(f"{__name__}.GraphManager")
        
        # Initialize workflow nodes
        self.content_analyzer = ContentAnalyzerNode(self.pipeline)
        self.risk_assessor = RiskAssessorNode()
        self.report_generator = ReportGeneratorNode()
        self.source_verifier = SourceVerifierNode()
        
        # Initialize workflows
        self.workflows = {}
        self._initialize_workflows()
        
        # Memory saver for checkpointing
        self.memory = MemorySaver()
    
    def _initialize_workflows(self):
        """Initialize all workflow graphs."""
        self.logger.info("Initializing LangGraph workflows")
        
        # Create workflow graphs
        self.workflows[WorkflowType.TEXT_ANALYSIS] = self._create_text_analysis_workflow()
        self.workflows[WorkflowType.IMAGE_ANALYSIS] = self._create_image_analysis_workflow()
        self.workflows[WorkflowType.URL_ANALYSIS] = self._create_url_analysis_workflow()
        
        self.logger.info("LangGraph workflows initialized successfully")
    
    def _create_text_analysis_workflow(self) -> StateGraph:
        """Create text analysis workflow with conditional routing."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("content_analyzer", self._content_analyzer_wrapper)
        workflow.add_node("risk_assessor", self._risk_assessor_wrapper)
        workflow.add_node("source_verifier", self._source_verifier_wrapper)
        workflow.add_node("report_generator", self._report_generator_wrapper)
        workflow.add_node("error_handler", self._error_handler_wrapper)
        
        # Set entry point
        workflow.set_entry_point("content_analyzer")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "content_analyzer",
            self._should_continue_after_content_analysis,
            {
                "continue": "source_verifier",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "source_verifier",
            self._should_continue_after_source_verification,
            {
                "continue": "risk_assessor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "risk_assessor",
            self._should_continue_after_risk_assessment,
            {
                "continue": "report_generator",
                "error": "error_handler"
            }
        )
        
        # Final edges
        workflow.add_edge("report_generator", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _create_image_analysis_workflow(self) -> StateGraph:
        """Create image analysis workflow with manipulation detection."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("content_analyzer", self._content_analyzer_wrapper)
        workflow.add_node("source_verifier", self._source_verifier_wrapper)
        workflow.add_node("risk_assessor", self._risk_assessor_wrapper)
        workflow.add_node("report_generator", self._report_generator_wrapper)
        workflow.add_node("error_handler", self._error_handler_wrapper)
        
        # Set entry point
        workflow.set_entry_point("content_analyzer")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "content_analyzer",
            self._should_continue_after_content_analysis,
            {
                "continue": "source_verifier",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "source_verifier",
            self._should_continue_after_source_verification,
            {
                "continue": "risk_assessor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "risk_assessor",
            self._should_continue_after_risk_assessment,
            {
                "continue": "report_generator",
                "error": "error_handler"
            }
        )
        
        # Final edges
        workflow.add_edge("report_generator", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _create_url_analysis_workflow(self) -> StateGraph:
        """Create URL analysis workflow with source verification."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("content_analyzer", self._content_analyzer_wrapper)
        workflow.add_node("source_verifier", self._source_verifier_wrapper)
        workflow.add_node("risk_assessor", self._risk_assessor_wrapper)
        workflow.add_node("report_generator", self._report_generator_wrapper)
        workflow.add_node("error_handler", self._error_handler_wrapper)
        
        # Set entry point
        workflow.set_entry_point("content_analyzer")
        
        # Add conditional edges - for URLs, source verification is critical
        workflow.add_conditional_edges(
            "content_analyzer",
            self._should_continue_after_content_analysis,
            {
                "continue": "source_verifier",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "source_verifier",
            self._should_continue_after_source_verification,
            {
                "continue": "risk_assessor",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "risk_assessor",
            self._should_continue_after_risk_assessment,
            {
                "continue": "report_generator",
                "error": "error_handler"
            }
        )
        
        # Final edges
        workflow.add_edge("report_generator", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def execute_workflow(
        self,
        content_type: ContentType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        workflow_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the appropriate workflow for the given content.
        
        Args:
            content_type: Type of content to analyze
            content: The content to analyze
            metadata: Additional metadata about the content
            workflow_config: Configuration for the workflow execution
            
        Returns:
            Dict containing the complete analysis results
        """
        try:
            self.logger.info(f"Starting workflow execution for {content_type.value} content")
            
            # Determine workflow type
            if content_type == ContentType.TEXT:
                workflow_type = WorkflowType.TEXT_ANALYSIS
            elif content_type == ContentType.IMAGE:
                workflow_type = WorkflowType.IMAGE_ANALYSIS
            elif content_type == ContentType.URL:
                workflow_type = WorkflowType.URL_ANALYSIS
            else:
                raise ContentProcessingError(f"Unsupported content type: {content_type}")
            
            # Get the appropriate workflow
            workflow = self.workflows[workflow_type]
            
            # Prepare initial state
            initial_state = GraphState(
                content_type=content_type.value,
                content=content,
                metadata=metadata or {},
                analysis_results={},
                error=None,
                processing_stage="initial",
                workflow_config=workflow_config or {}
            )
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"{workflow_type.value}_{asyncio.current_task().get_name()}"}}
            
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Log completion
            if final_state.get("error"):
                self.logger.error(f"Workflow completed with error: {final_state['error']}")
            else:
                self.logger.info(f"Workflow completed successfully: {final_state['processing_stage']}")
            
            return final_state
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "content_type": content_type.value,
                "content": content,
                "metadata": metadata or {},
                "analysis_results": {},
                "error": f"Workflow execution failed: {str(e)}",
                "processing_stage": "workflow_error",
                "workflow_config": workflow_config or {}
            }
    
    # Node wrapper functions to convert between state formats
    async def _content_analyzer_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for content analyzer node."""
        try:
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType(state["content_type"]),
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Execute content analysis
            result_state = await self.content_analyzer.analyze_content(analysis_state)
            
            # Convert back to GraphState
            return GraphState(
                content_type=result_state.content_type.value,
                content=result_state.content,
                metadata=result_state.metadata,
                analysis_results=result_state.analysis_results,
                error=result_state.error,
                processing_stage=result_state.processing_stage,
                workflow_config=state["workflow_config"]
            )
            
        except Exception as e:
            self.logger.error(f"Content analyzer wrapper failed: {str(e)}")
            state["error"] = f"Content analysis failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _risk_assessor_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for risk assessor node."""
        try:
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType(state["content_type"]),
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Execute risk assessment
            result_state = await self.risk_assessor.assess_risk(analysis_state)
            
            # Convert back to GraphState
            return GraphState(
                content_type=result_state.content_type.value,
                content=result_state.content,
                metadata=result_state.metadata,
                analysis_results=result_state.analysis_results,
                error=result_state.error,
                processing_stage=result_state.processing_stage,
                workflow_config=state["workflow_config"]
            )
            
        except Exception as e:
            self.logger.error(f"Risk assessor wrapper failed: {str(e)}")
            state["error"] = f"Risk assessment failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _report_generator_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for report generator node."""
        try:
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType(state["content_type"]),
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Execute report generation
            result_state = await self.report_generator.generate_report(analysis_state)
            
            # Convert back to GraphState
            return GraphState(
                content_type=result_state.content_type.value,
                content=result_state.content,
                metadata=result_state.metadata,
                analysis_results=result_state.analysis_results,
                error=result_state.error,
                processing_stage=result_state.processing_stage,
                workflow_config=state["workflow_config"]
            )
            
        except Exception as e:
            self.logger.error(f"Report generator wrapper failed: {str(e)}")
            state["error"] = f"Report generation failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _source_verifier_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for source verifier node."""
        try:
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType(state["content_type"]),
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Execute source verification
            result_state = await self.source_verifier.verify_sources(analysis_state)
            
            # Convert back to GraphState
            return GraphState(
                content_type=result_state.content_type.value,
                content=result_state.content,
                metadata=result_state.metadata,
                analysis_results=result_state.analysis_results,
                error=result_state.error,
                processing_stage=result_state.processing_stage,
                workflow_config=state["workflow_config"]
            )
            
        except Exception as e:
            self.logger.error(f"Source verifier wrapper failed: {str(e)}")
            state["error"] = f"Source verification failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _error_handler_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for error handling."""
        self.logger.error(f"Workflow error handler activated: {state.get('error', 'Unknown error')}")
        
        # Ensure we have a minimal analysis result even in error cases
        if not state["analysis_results"]:
            state["analysis_results"] = {
                "error_report": {
                    "error_message": state.get("error", "Unknown error occurred"),
                    "processing_stage": state.get("processing_stage", "unknown"),
                    "content_type": state.get("content_type", "unknown"),
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
        
        state["processing_stage"] = "error_handled"
        return state
    
    # Conditional routing functions
    def _should_continue_after_content_analysis(self, state: GraphState) -> str:
        """Determine next step after content analysis."""
        if state.get("error"):
            return "error"
        
        processing_stage = state.get("processing_stage", "")
        if processing_stage == "content_analysis_complete":
            return "continue"
        
        return "error"
    
    def _should_continue_after_source_verification(self, state: GraphState) -> str:
        """Determine next step after source verification."""
        if state.get("error"):
            return "error"
        
        processing_stage = state.get("processing_stage", "")
        if processing_stage == "source_verification_complete":
            return "continue"
        
        return "error"
    
    def _should_continue_after_risk_assessment(self, state: GraphState) -> str:
        """Determine next step after risk assessment."""
        if state.get("error"):
            return "error"
        
        processing_stage = state.get("processing_stage", "")
        if processing_stage == "risk_assessment_complete":
            return "continue"
        
        return "error"
    
    async def get_workflow_status(self, thread_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow execution."""
        try:
            # This would retrieve the current state from the checkpointer
            # For now, return a basic status
            return {
                "thread_id": thread_id,
                "status": "unknown",
                "message": "Status retrieval not fully implemented"
            }
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {str(e)}")
            return {
                "thread_id": thread_id,
                "status": "error",
                "message": f"Failed to retrieve status: {str(e)}"
            }
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow types."""
        return [workflow_type.value for workflow_type in WorkflowType]
    
    async def validate_workflow_input(
        self,
        content_type: ContentType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate input before workflow execution.
        
        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []
        
        # Basic content validation
        if not content or not content.strip():
            errors.append("Content cannot be empty")
        
        # Content type specific validation
        if content_type == ContentType.TEXT:
            if len(content) > 10000:  # 10k character limit
                errors.append("Text content exceeds maximum length (10,000 characters)")
        
        elif content_type == ContentType.URL:
            if not content.startswith(('http://', 'https://')):
                errors.append("URL must start with http:// or https://")
        
        elif content_type == ContentType.IMAGE:
            # For images, content would be description or path
            if len(content) < 10:
                errors.append("Image description too short")
        
        # Metadata validation
        if metadata:
            if not isinstance(metadata, dict):
                errors.append("Metadata must be a dictionary")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }