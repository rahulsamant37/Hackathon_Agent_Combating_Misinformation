"""
Text analysis workflow with conditional routing.

This module defines the specific workflow for analyzing text content,
including conditional routing based on analysis results and error recovery.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.nodes.content_analyzer import ContentAnalyzerNode, ContentAnalysisState, ContentType
from graph.nodes.risk_assessor import RiskAssessorNode
from graph.nodes.report_generator import ReportGeneratorNode
from graph.nodes.source_verifier import SourceVerifierNode
from llm.langchain_pipeline import LangChainPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class TextAnalysisWorkflow:
    """
    Specialized workflow for text content analysis.
    
    Provides enhanced text-specific routing and processing logic
    with conditional paths based on content characteristics.
    """
    
    def __init__(self, pipeline: LangChainPipeline):
        self.pipeline = pipeline
        self.logger = get_logger(f"{__name__}.TextAnalysisWorkflow")
        
        # Initialize nodes
        self.content_analyzer = ContentAnalyzerNode(pipeline)
        self.risk_assessor = RiskAssessorNode()
        self.report_generator = ReportGeneratorNode()
        self.source_verifier = SourceVerifierNode()
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the text analysis workflow graph."""
        from graph.graph_manager import GraphState
        
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("check_fact_requirements", self._check_fact_requirements_node)
        workflow.add_node("verify_sources", self._verify_sources_node)
        workflow.add_node("assess_risk", self._assess_risk_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_content")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "analyze_content",
            self._route_after_content_analysis,
            {
                "fact_check_required": "check_fact_requirements",
                "direct_to_sources": "verify_sources",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "check_fact_requirements",
            self._route_after_fact_check,
            {
                "verify_sources": "verify_sources",
                "skip_sources": "assess_risk",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "verify_sources",
            self._route_after_source_verification,
            {
                "assess_risk": "assess_risk",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "assess_risk",
            self._route_after_risk_assessment,
            {
                "generate_report": "generate_report",
                "error": "handle_error"
            }
        )
        
        # Final edges
        workflow.add_edge("generate_report", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def _analyze_content_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text content with enhanced text-specific processing."""
        try:
            self.logger.info("Starting text content analysis")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.TEXT,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform content analysis
            result_state = await self.content_analyzer.analyze_content(analysis_state)
            
            # Add text-specific enhancements
            if not result_state.error:
                result_state = await self._enhance_text_analysis(result_state)
            
            # Update state
            state.update({
                "analysis_results": result_state.analysis_results,
                "error": result_state.error,
                "processing_stage": result_state.processing_stage
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Text content analysis failed: {str(e)}")
            state["error"] = f"Text analysis failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _check_fact_requirements_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if content requires fact-checking and prepare accordingly."""
        try:
            self.logger.info("Checking fact-checking requirements")
            
            content_analysis = state["analysis_results"].get("content_analysis", {})
            requires_fact_check = content_analysis.get("requires_fact_check", False)
            
            # Enhanced fact-checking requirement analysis
            fact_check_priority = self._assess_fact_check_priority(state["content"], content_analysis)
            
            # Update analysis results with fact-checking information
            state["analysis_results"]["fact_check_assessment"] = {
                "requires_fact_check": requires_fact_check,
                "priority": fact_check_priority,
                "recommended_sources": self._get_recommended_fact_check_sources(state["content"]),
                "checkable_claims": self._extract_checkable_claims(state["content"])
            }
            
            state["processing_stage"] = "fact_check_requirements_assessed"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Fact-check requirements assessment failed: {str(e)}")
            state["error"] = f"Fact-check assessment failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _verify_sources_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify sources with text-specific enhancements."""
        try:
            self.logger.info("Verifying sources for text content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.TEXT,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform source verification
            result_state = await self.source_verifier.verify_sources(analysis_state)
            
            # Add text-specific source analysis
            if not result_state.error:
                result_state = await self._enhance_source_verification(result_state)
            
            # Update state
            state.update({
                "analysis_results": result_state.analysis_results,
                "error": result_state.error,
                "processing_stage": result_state.processing_stage
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Source verification failed: {str(e)}")
            state["error"] = f"Source verification failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _assess_risk_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk with text-specific considerations."""
        try:
            self.logger.info("Assessing risk for text content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.TEXT,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform risk assessment
            result_state = await self.risk_assessor.assess_risk(analysis_state)
            
            # Add text-specific risk adjustments
            if not result_state.error:
                result_state = await self._enhance_risk_assessment(result_state)
            
            # Update state
            state.update({
                "analysis_results": result_state.analysis_results,
                "error": result_state.error,
                "processing_stage": result_state.processing_stage
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            state["error"] = f"Risk assessment failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _generate_report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report with text-specific formatting."""
        try:
            self.logger.info("Generating report for text content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.TEXT,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Generate report
            result_state = await self.report_generator.generate_report(analysis_state)
            
            # Add text-specific report enhancements
            if not result_state.error:
                result_state = await self._enhance_report_generation(result_state)
            
            # Update state
            state.update({
                "analysis_results": result_state.analysis_results,
                "error": result_state.error,
                "processing_stage": result_state.processing_stage
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            state["error"] = f"Report generation failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _handle_error_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors with text-specific recovery."""
        self.logger.error(f"Text workflow error: {state.get('error', 'Unknown error')}")
        
        # Create minimal error report
        error_report = {
            "error_type": "text_analysis_error",
            "error_message": state.get("error", "Unknown error"),
            "content_preview": state.get("content", "")[:100] + "..." if len(state.get("content", "")) > 100 else state.get("content", ""),
            "recovery_suggestions": [
                "Try with shorter text content",
                "Check for special characters or encoding issues",
                "Verify text is in a supported language"
            ]
        }
        
        state["analysis_results"]["error_report"] = error_report
        state["processing_stage"] = "error_handled"
        
        return state
    
    # Routing functions
    def _route_after_content_analysis(self, state: Dict[str, Any]) -> str:
        """Route after content analysis based on results."""
        if state.get("error"):
            return "error"
        
        content_analysis = state["analysis_results"].get("content_analysis", {})
        
        # Check if fact-checking is required
        if content_analysis.get("requires_fact_check", False):
            return "fact_check_required"
        
        # Check if there are URLs to verify
        text_metrics = content_analysis.get("text_metrics", {})
        if text_metrics.get("has_urls", False):
            return "direct_to_sources"
        
        return "direct_to_sources"  # Always verify sources for completeness
    
    def _route_after_fact_check(self, state: Dict[str, Any]) -> str:
        """Route after fact-check requirements assessment."""
        if state.get("error"):
            return "error"
        
        fact_check_assessment = state["analysis_results"].get("fact_check_assessment", {})
        priority = fact_check_assessment.get("priority", "low")
        
        # High priority fact-checking requires source verification
        if priority in ["high", "critical"]:
            return "verify_sources"
        
        # Low priority can skip detailed source verification
        return "skip_sources"
    
    def _route_after_source_verification(self, state: Dict[str, Any]) -> str:
        """Route after source verification."""
        if state.get("error"):
            return "error"
        
        return "assess_risk"
    
    def _route_after_risk_assessment(self, state: Dict[str, Any]) -> str:
        """Route after risk assessment."""
        if state.get("error"):
            return "error"
        
        return "generate_report"
    
    # Enhancement functions
    async def _enhance_text_analysis(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add text-specific analysis enhancements."""
        content_analysis = state.analysis_results.get("content_analysis", {})
        
        # Add readability analysis
        readability_score = self._calculate_readability_score(state.content)
        content_analysis["readability_score"] = readability_score
        
        # Add sentiment analysis indicators
        sentiment_indicators = self._analyze_sentiment_indicators(state.content)
        content_analysis["sentiment_indicators"] = sentiment_indicators
        
        # Add linguistic complexity analysis
        complexity_analysis = self._analyze_linguistic_complexity(state.content)
        content_analysis["complexity_analysis"] = complexity_analysis
        
        state.analysis_results["content_analysis"] = content_analysis
        return state
    
    async def _enhance_source_verification(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add text-specific source verification enhancements."""
        source_verification = state.analysis_results.get("source_verification", {})
        
        # Add citation analysis
        citation_analysis = self._analyze_citations(state.content)
        source_verification["citation_analysis"] = citation_analysis
        
        # Add authority indicators
        authority_indicators = self._identify_authority_indicators(state.content)
        source_verification["authority_indicators"] = authority_indicators
        
        state.analysis_results["source_verification"] = source_verification
        return state
    
    async def _enhance_risk_assessment(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add text-specific risk assessment enhancements."""
        risk_assessment = state.analysis_results.get("risk_assessment", {})
        
        # Adjust risk based on text-specific factors
        fact_check_assessment = state.analysis_results.get("fact_check_assessment", {})
        if fact_check_assessment.get("priority") == "critical":
            # Increase risk for critical fact-checking needs
            current_score = risk_assessment.get("risk_score", 0.5)
            risk_assessment["risk_score"] = min(1.0, current_score + 0.1)
            risk_assessment["text_specific_adjustments"] = ["Critical fact-checking required"]
        
        state.analysis_results["risk_assessment"] = risk_assessment
        return state
    
    async def _enhance_report_generation(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add text-specific report enhancements."""
        final_report = state.analysis_results.get("final_report", {})
        
        # Add text-specific recommendations
        text_recommendations = self._generate_text_specific_recommendations(state)
        final_report["text_specific_recommendations"] = text_recommendations
        
        # Add readability assessment
        content_analysis = state.analysis_results.get("content_analysis", {})
        readability_score = content_analysis.get("readability_score", 0.5)
        final_report["readability_assessment"] = {
            "score": readability_score,
            "interpretation": self._interpret_readability_score(readability_score)
        }
        
        state.analysis_results["final_report"] = final_report
        return state
    
    # Helper functions
    def _assess_fact_check_priority(self, content: str, content_analysis: Dict[str, Any]) -> str:
        """Assess the priority level for fact-checking."""
        priority_score = 0
        
        # Check for high-priority keywords
        high_priority_keywords = [
            "study shows", "research proves", "scientists say", "data reveals",
            "statistics show", "according to experts", "new study", "breakthrough"
        ]
        
        content_lower = content.lower()
        for keyword in high_priority_keywords:
            if keyword in content_lower:
                priority_score += 2
        
        # Check for numbers and percentages
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?%?', content)
        priority_score += min(len(numbers), 5)  # Cap at 5 points
        
        # Check indicators from content analysis
        indicators = content_analysis.get("initial_indicators", [])
        factual_indicators = [ind for ind in indicators if "factual" in ind.get("type", "").lower()]
        priority_score += len(factual_indicators) * 2
        
        # Determine priority level
        if priority_score >= 10:
            return "critical"
        elif priority_score >= 6:
            return "high"
        elif priority_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_fact_check_sources(self, content: str) -> List[str]:
        """Get recommended fact-checking sources based on content."""
        sources = ["https://www.factcheck.org", "https://www.snopes.com"]
        
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["politics", "election", "government"]):
            sources.append("https://www.politifact.com")
        
        if any(word in content_lower for word in ["health", "medical", "vaccine", "covid"]):
            sources.extend(["https://www.who.int", "https://www.cdc.gov"])
        
        if any(word in content_lower for word in ["climate", "environment", "global warming"]):
            sources.append("https://climate.nasa.gov")
        
        return sources
    
    def _extract_checkable_claims(self, content: str) -> List[str]:
        """Extract specific claims that can be fact-checked."""
        claims = []
        
        # Simple pattern matching for factual claims
        import re
        
        # Look for sentences with statistical claims
        stat_patterns = [
            r'[^.]*\d+(?:\.\d+)?%[^.]*\.',
            r'[^.]*\d+(?:,\d{3})*\s+(?:people|cases|deaths|studies)[^.]*\.',
            r'[^.]*(?:increased|decreased|rose|fell)\s+by\s+\d+[^.]*\.'
        ]
        
        for pattern in stat_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches[:3])  # Limit to 3 per pattern
        
        return claims[:5]  # Total limit of 5 claims
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score."""
        words = content.split()
        sentences = content.count('.') + content.count('!') + content.count('?')
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = len(words) / sentences
        
        # Simple readability heuristic (lower is more readable)
        if avg_words_per_sentence <= 15:
            return 0.8  # High readability
        elif avg_words_per_sentence <= 25:
            return 0.6  # Medium readability
        else:
            return 0.3  # Low readability
    
    def _analyze_sentiment_indicators(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment indicators in the text."""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disgusting", "outrageous"]
        emotional_words = ["shocking", "unbelievable", "incredible", "devastating", "alarming"]
        
        content_lower = content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        emotional_count = sum(1 for word in emotional_words if word in content_lower)
        
        return {
            "positive_sentiment": positive_count,
            "negative_sentiment": negative_count,
            "emotional_language": emotional_count,
            "sentiment_ratio": (positive_count - negative_count) / max(1, positive_count + negative_count)
        }
    
    def _analyze_linguistic_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze linguistic complexity of the text."""
        words = content.split()
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Count complex words (more than 6 characters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_word_ratio = complex_words / len(words) if words else 0
        
        return {
            "average_word_length": avg_word_length,
            "complex_word_ratio": complex_word_ratio,
            "total_words": len(words),
            "complexity_score": (avg_word_length / 10) + complex_word_ratio
        }
    
    def _analyze_citations(self, content: str) -> Dict[str, Any]:
        """Analyze citation patterns in the text."""
        import re
        
        # Look for citation patterns
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[[^\]]*\d+[^\]]*\]',  # [1], [Smith et al.]
            r'according to [^,\.]+',  # according to X
            r'as reported by [^,\.]+' # as reported by X
        ]
        
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, content, re.IGNORECASE))
        
        return {
            "citations_found": citations_found,
            "has_formal_citations": citations_found > 0,
            "citation_density": citations_found / len(content.split()) if content.split() else 0
        }
    
    def _identify_authority_indicators(self, content: str) -> List[str]:
        """Identify indicators of authority in the text."""
        authority_indicators = []
        
        authority_phrases = [
            "peer-reviewed", "published in", "journal", "university", "professor",
            "doctor", "phd", "research", "study", "clinical trial", "meta-analysis"
        ]
        
        content_lower = content.lower()
        for phrase in authority_phrases:
            if phrase in content_lower:
                authority_indicators.append(phrase)
        
        return authority_indicators
    
    def _generate_text_specific_recommendations(self, state: ContentAnalysisState) -> List[str]:
        """Generate text-specific recommendations."""
        recommendations = []
        
        content_analysis = state.analysis_results.get("content_analysis", {})
        
        # Readability recommendations
        readability_score = content_analysis.get("readability_score", 0.5)
        if readability_score < 0.4:
            recommendations.append("Consider the complexity of the language - simpler writing is often more trustworthy")
        
        # Sentiment recommendations
        sentiment_indicators = content_analysis.get("sentiment_indicators", {})
        if sentiment_indicators.get("emotional_language", 0) > 3:
            recommendations.append("Be cautious of highly emotional language that may be designed to manipulate")
        
        # Citation recommendations
        source_verification = state.analysis_results.get("source_verification", {})
        citation_analysis = source_verification.get("citation_analysis", {})
        if not citation_analysis.get("has_formal_citations", False):
            recommendations.append("Look for proper citations and references to verify claims")
        
        return recommendations
    
    def _interpret_readability_score(self, score: float) -> str:
        """Interpret readability score."""
        if score >= 0.7:
            return "Highly readable - clear and accessible language"
        elif score >= 0.5:
            return "Moderately readable - some complex language"
        else:
            return "Low readability - complex or potentially obfuscated language"
    
    async def execute(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the text analysis workflow."""
        from graph.graph_manager import GraphState
        
        initial_state = GraphState(
            content_type="text",
            content=content,
            metadata=metadata or {},
            analysis_results={},
            error=None,
            processing_stage="initial",
            workflow_config=config or {}
        )
        
        # Execute workflow
        thread_config = {"configurable": {"thread_id": f"text_{asyncio.current_task().get_name()}"}}
        final_state = await self.workflow.ainvoke(initial_state, config=thread_config)
        
        return final_state