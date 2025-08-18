"""
Image analysis workflow with manipulation detection.

This module defines the specific workflow for analyzing image content,
including manipulation detection and reverse image search integration.
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


class ImageAnalysisWorkflow:
    """
    Specialized workflow for image content analysis.
    
    Provides enhanced image-specific processing including manipulation detection,
    reverse image search integration, and technical analysis routing.
    """
    
    def __init__(self, pipeline: LangChainPipeline):
        self.pipeline = pipeline
        self.logger = get_logger(f"{__name__}.ImageAnalysisWorkflow")
        
        # Initialize nodes
        self.content_analyzer = ContentAnalyzerNode(pipeline)
        self.risk_assessor = RiskAssessorNode()
        self.report_generator = ReportGeneratorNode()
        self.source_verifier = SourceVerifierNode()
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the image analysis workflow graph."""
        from graph.graph_manager import GraphState
        
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyze_image", self._analyze_image_node)
        workflow.add_node("technical_analysis", self._technical_analysis_node)
        workflow.add_node("reverse_search", self._reverse_search_node)
        workflow.add_node("verify_sources", self._verify_sources_node)
        workflow.add_node("assess_risk", self._assess_risk_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_image")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "analyze_image",
            self._route_after_image_analysis,
            {
                "technical_analysis": "technical_analysis",
                "reverse_search": "reverse_search",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "technical_analysis",
            self._route_after_technical_analysis,
            {
                "reverse_search": "reverse_search",
                "verify_sources": "verify_sources",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "reverse_search",
            self._route_after_reverse_search,
            {
                "verify_sources": "verify_sources",
                "assess_risk": "assess_risk",
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
    
    async def _analyze_image_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image content with enhanced image-specific processing."""
        try:
            self.logger.info("Starting image content analysis")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.IMAGE,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform content analysis
            result_state = await self.content_analyzer.analyze_content(analysis_state)
            
            # Add image-specific enhancements
            if not result_state.error:
                result_state = await self._enhance_image_analysis(result_state)
            
            # Update state
            state.update({
                "analysis_results": result_state.analysis_results,
                "error": result_state.error,
                "processing_stage": result_state.processing_stage
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Image content analysis failed: {str(e)}")
            state["error"] = f"Image analysis failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _technical_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed technical analysis of the image."""
        try:
            self.logger.info("Performing technical image analysis")
            
            # Simulate technical analysis (in real implementation, this would use image processing libraries)
            technical_analysis = await self._perform_technical_analysis(
                state["content"], 
                state["metadata"]
            )
            
            # Update analysis results
            state["analysis_results"]["technical_analysis"] = technical_analysis
            state["processing_stage"] = "technical_analysis_complete"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {str(e)}")
            state["error"] = f"Technical analysis failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _reverse_search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reverse image search and source identification."""
        try:
            self.logger.info("Performing reverse image search")
            
            # Simulate reverse image search (in real implementation, this would use actual search APIs)
            reverse_search_results = await self._perform_reverse_search(
                state["content"],
                state["metadata"]
            )
            
            # Update analysis results
            state["analysis_results"]["reverse_search"] = reverse_search_results
            state["processing_stage"] = "reverse_search_complete"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Reverse search failed: {str(e)}")
            state["error"] = f"Reverse search failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _verify_sources_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify sources with image-specific enhancements."""
        try:
            self.logger.info("Verifying sources for image content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.IMAGE,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform source verification
            result_state = await self.source_verifier.verify_sources(analysis_state)
            
            # Add image-specific source analysis
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
        """Assess risk with image-specific considerations."""
        try:
            self.logger.info("Assessing risk for image content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.IMAGE,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform risk assessment
            result_state = await self.risk_assessor.assess_risk(analysis_state)
            
            # Add image-specific risk adjustments
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
        """Generate report with image-specific formatting."""
        try:
            self.logger.info("Generating report for image content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.IMAGE,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Generate report
            result_state = await self.report_generator.generate_report(analysis_state)
            
            # Add image-specific report enhancements
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
        """Handle errors with image-specific recovery."""
        self.logger.error(f"Image workflow error: {state.get('error', 'Unknown error')}")
        
        # Create minimal error report
        error_report = {
            "error_type": "image_analysis_error",
            "error_message": state.get("error", "Unknown error"),
            "image_info": {
                "description": state.get("content", "")[:100] + "..." if len(state.get("content", "")) > 100 else state.get("content", ""),
                "metadata_available": bool(state.get("metadata"))
            },
            "recovery_suggestions": [
                "Try with a different image format",
                "Ensure image metadata is available",
                "Check image file size and quality",
                "Verify image is not corrupted"
            ]
        }
        
        state["analysis_results"]["error_report"] = error_report
        state["processing_stage"] = "error_handled"
        
        return state
    
    # Routing functions
    def _route_after_image_analysis(self, state: Dict[str, Any]) -> str:
        """Route after image analysis based on results."""
        if state.get("error"):
            return "error"
        
        content_analysis = state["analysis_results"].get("content_analysis", {})
        
        # Check if technical analysis is required
        if content_analysis.get("requires_technical_analysis", False):
            return "technical_analysis"
        
        # Otherwise go directly to reverse search
        return "reverse_search"
    
    def _route_after_technical_analysis(self, state: Dict[str, Any]) -> str:
        """Route after technical analysis."""
        if state.get("error"):
            return "error"
        
        technical_analysis = state["analysis_results"].get("technical_analysis", {})
        
        # If manipulation is detected, prioritize source verification
        if technical_analysis.get("manipulation_detected", False):
            return "verify_sources"
        
        # Otherwise proceed to reverse search
        return "reverse_search"
    
    def _route_after_reverse_search(self, state: Dict[str, Any]) -> str:
        """Route after reverse search."""
        if state.get("error"):
            return "error"
        
        reverse_search = state["analysis_results"].get("reverse_search", {})
        
        # If sources were found, verify them
        if reverse_search.get("sources_found", 0) > 0:
            return "verify_sources"
        
        # If no sources found, proceed directly to risk assessment
        return "assess_risk"
    
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
    async def _enhance_image_analysis(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add image-specific analysis enhancements."""
        content_analysis = state.analysis_results.get("content_analysis", {})
        
        # Add EXIF analysis
        exif_analysis = self._analyze_exif_data(state.metadata)
        content_analysis["exif_analysis"] = exif_analysis
        
        # Add visual content analysis
        visual_analysis = self._analyze_visual_content(state.content, state.metadata)
        content_analysis["visual_analysis"] = visual_analysis
        
        # Add compression analysis
        compression_analysis = self._analyze_compression_artifacts(state.metadata)
        content_analysis["compression_analysis"] = compression_analysis
        
        state.analysis_results["content_analysis"] = content_analysis
        return state
    
    async def _perform_technical_analysis(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed technical analysis of the image."""
        # Simulate technical analysis results
        # In a real implementation, this would use image processing libraries like OpenCV, PIL, etc.
        
        image_metrics = metadata.get("image_metrics", {})
        
        analysis_results = {
            "manipulation_detected": False,
            "manipulation_confidence": 0.0,
            "technical_indicators": [],
            "forensic_analysis": {},
            "quality_assessment": {}
        }
        
        # Simulate manipulation detection based on metadata
        file_size = image_metrics.get("file_size", 0)
        has_exif = image_metrics.get("has_exif", False)
        
        # Heuristic: Very small files or missing EXIF might indicate manipulation
        if file_size < 50000 and not has_exif:  # Less than 50KB and no EXIF
            analysis_results["manipulation_detected"] = True
            analysis_results["manipulation_confidence"] = 0.7
            analysis_results["technical_indicators"].append("Suspicious file size and missing EXIF data")
        
        # Simulate compression analysis
        compression_ratio = image_metrics.get("compression_ratio", 0)
        if compression_ratio > 0.9:  # Very high compression
            analysis_results["technical_indicators"].append("High compression ratio detected")
        
        # Simulate quality assessment
        dimensions = image_metrics.get("dimensions", {})
        width = dimensions.get("width", 0)
        height = dimensions.get("height", 0)
        
        if width > 0 and height > 0:
            aspect_ratio = width / height
            analysis_results["quality_assessment"] = {
                "resolution": f"{width}x{height}",
                "aspect_ratio": aspect_ratio,
                "pixel_count": width * height,
                "quality_score": min(1.0, (width * height) / 1000000)  # Normalize by 1MP
            }
        
        return analysis_results
    
    async def _perform_reverse_search(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reverse image search simulation."""
        # Simulate reverse image search results
        # In a real implementation, this would use Google Images API, TinEye API, etc.
        
        import random
        
        # Simulate finding sources
        sources_found = random.randint(0, 10)
        
        search_results = {
            "sources_found": sources_found,
            "search_engines_used": ["Google Images", "TinEye", "Yandex"],
            "earliest_occurrence": None,
            "source_urls": [],
            "similar_images": sources_found,
            "exact_matches": max(0, sources_found - 3)
        }
        
        # Generate simulated source URLs
        if sources_found > 0:
            domains = [
                "reuters.com", "ap.org", "bbc.com", "cnn.com", "nytimes.com",
                "facebook.com", "twitter.com", "instagram.com", "reddit.com"
            ]
            
            for i in range(min(sources_found, 5)):
                domain = random.choice(domains)
                search_results["source_urls"].append(f"https://{domain}/image_{i}")
            
            # Simulate earliest occurrence
            search_results["earliest_occurrence"] = "2023-01-15"
        
        return search_results
    
    async def _enhance_source_verification(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add image-specific source verification enhancements."""
        source_verification = state.analysis_results.get("source_verification", {})
        
        # Add reverse search correlation
        reverse_search = state.analysis_results.get("reverse_search", {})
        source_correlation = self._correlate_sources_with_search(
            source_verification, reverse_search
        )
        source_verification["source_correlation"] = source_correlation
        
        # Add temporal analysis
        temporal_analysis = self._analyze_temporal_consistency(reverse_search)
        source_verification["temporal_analysis"] = temporal_analysis
        
        state.analysis_results["source_verification"] = source_verification
        return state
    
    async def _enhance_risk_assessment(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add image-specific risk assessment enhancements."""
        risk_assessment = state.analysis_results.get("risk_assessment", {})
        
        # Adjust risk based on technical analysis
        technical_analysis = state.analysis_results.get("technical_analysis", {})
        if technical_analysis.get("manipulation_detected", False):
            current_score = risk_assessment.get("risk_score", 0.5)
            manipulation_confidence = technical_analysis.get("manipulation_confidence", 0.5)
            risk_assessment["risk_score"] = min(1.0, current_score + (manipulation_confidence * 0.3))
            risk_assessment["image_specific_adjustments"] = ["Technical manipulation detected"]
        
        # Adjust risk based on reverse search results
        reverse_search = state.analysis_results.get("reverse_search", {})
        if reverse_search.get("sources_found", 0) == 0:
            current_score = risk_assessment.get("risk_score", 0.5)
            risk_assessment["risk_score"] = min(1.0, current_score + 0.2)
            adjustments = risk_assessment.get("image_specific_adjustments", [])
            adjustments.append("No sources found in reverse search")
            risk_assessment["image_specific_adjustments"] = adjustments
        
        state.analysis_results["risk_assessment"] = risk_assessment
        return state
    
    async def _enhance_report_generation(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add image-specific report enhancements."""
        final_report = state.analysis_results.get("final_report", {})
        
        # Add image-specific recommendations
        image_recommendations = self._generate_image_specific_recommendations(state)
        final_report["image_specific_recommendations"] = image_recommendations
        
        # Add technical summary
        technical_analysis = state.analysis_results.get("technical_analysis", {})
        final_report["technical_summary"] = {
            "manipulation_detected": technical_analysis.get("manipulation_detected", False),
            "confidence": technical_analysis.get("manipulation_confidence", 0.0),
            "key_indicators": technical_analysis.get("technical_indicators", [])
        }
        
        # Add reverse search summary
        reverse_search = state.analysis_results.get("reverse_search", {})
        final_report["reverse_search_summary"] = {
            "sources_found": reverse_search.get("sources_found", 0),
            "earliest_occurrence": reverse_search.get("earliest_occurrence"),
            "verification_status": "verified" if reverse_search.get("sources_found", 0) > 0 else "unverified"
        }
        
        state.analysis_results["final_report"] = final_report
        return state
    
    # Helper functions
    def _analyze_exif_data(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze EXIF data for authenticity indicators."""
        exif_data = metadata.get("exif_data", {})
        
        analysis = {
            "has_exif": bool(exif_data),
            "camera_info": {},
            "timestamp_info": {},
            "location_info": {},
            "authenticity_indicators": []
        }
        
        if exif_data:
            # Camera information
            if "Make" in exif_data or "Model" in exif_data:
                analysis["camera_info"] = {
                    "make": exif_data.get("Make", "Unknown"),
                    "model": exif_data.get("Model", "Unknown"),
                    "software": exif_data.get("Software", "Unknown")
                }
                analysis["authenticity_indicators"].append("Camera information present")
            
            # Timestamp information
            if "DateTime" in exif_data:
                analysis["timestamp_info"] = {
                    "creation_date": exif_data.get("DateTime"),
                    "has_timestamp": True
                }
                analysis["authenticity_indicators"].append("Original timestamp present")
            
            # GPS information
            if any(key.startswith("GPS") for key in exif_data.keys()):
                analysis["location_info"]["has_gps"] = True
                analysis["authenticity_indicators"].append("GPS location data present")
        else:
            analysis["authenticity_indicators"].append("No EXIF data - potentially processed")
        
        return analysis
    
    def _analyze_visual_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual content characteristics."""
        # Simulate visual content analysis
        # In real implementation, this would use computer vision techniques
        
        return {
            "content_type": "photograph",  # Could be: photograph, digital_art, screenshot, etc.
            "scene_analysis": {
                "indoor_outdoor": "unknown",
                "lighting_conditions": "natural",
                "people_detected": False,
                "objects_detected": []
            },
            "visual_quality": {
                "sharpness": 0.7,
                "noise_level": 0.3,
                "color_balance": 0.8,
                "overall_quality": 0.6
            },
            "anomaly_detection": {
                "inconsistent_lighting": False,
                "unnatural_shadows": False,
                "edge_artifacts": False,
                "color_inconsistencies": False
            }
        }
    
    def _analyze_compression_artifacts(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compression artifacts that might indicate manipulation."""
        image_metrics = metadata.get("image_metrics", {})
        
        return {
            "compression_level": image_metrics.get("compression_ratio", 0.5),
            "format": image_metrics.get("format", "unknown"),
            "quality_estimate": 0.7,  # Simulated quality estimate
            "recompression_detected": False,  # Would be detected through analysis
            "artifacts_present": [],
            "compression_consistency": True
        }
    
    def _correlate_sources_with_search(
        self, 
        source_verification: Dict[str, Any], 
        reverse_search: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate source verification with reverse search results."""
        source_urls = reverse_search.get("source_urls", [])
        reputation_indicators = source_verification.get("reputation_indicators", [])
        
        correlation = {
            "sources_match_search": False,
            "credible_sources_found": 0,
            "social_media_sources": 0,
            "news_sources": 0,
            "unknown_sources": 0
        }
        
        # Analyze source types
        for url in source_urls:
            if any(domain in url for domain in ["facebook.com", "twitter.com", "instagram.com"]):
                correlation["social_media_sources"] += 1
            elif any(domain in url for domain in ["reuters.com", "ap.org", "bbc.com", "cnn.com"]):
                correlation["news_sources"] += 1
                correlation["credible_sources_found"] += 1
            else:
                correlation["unknown_sources"] += 1
        
        # Check if verification matches search results
        if reputation_indicators and source_urls:
            correlation["sources_match_search"] = True
        
        return correlation
    
    def _analyze_temporal_consistency(self, reverse_search: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal consistency of image appearances."""
        earliest_occurrence = reverse_search.get("earliest_occurrence")
        sources_found = reverse_search.get("sources_found", 0)
        
        analysis = {
            "earliest_known_date": earliest_occurrence,
            "temporal_spread": "unknown",
            "consistency_score": 0.5,
            "red_flags": []
        }
        
        if earliest_occurrence:
            # Simulate temporal analysis
            from datetime import datetime, timedelta
            
            try:
                earliest_date = datetime.strptime(earliest_occurrence, "%Y-%m-%d")
                current_date = datetime.now()
                age_days = (current_date - earliest_date).days
                
                if age_days < 7:
                    analysis["red_flags"].append("Very recent image - limited verification time")
                    analysis["consistency_score"] = 0.3
                elif age_days > 365:
                    analysis["consistency_score"] = 0.8
                else:
                    analysis["consistency_score"] = 0.6
                
                analysis["temporal_spread"] = f"{age_days} days"
                
            except ValueError:
                analysis["red_flags"].append("Invalid date format in search results")
        
        return analysis
    
    def _generate_image_specific_recommendations(self, state: ContentAnalysisState) -> List[str]:
        """Generate image-specific recommendations."""
        recommendations = []
        
        # Technical analysis recommendations
        technical_analysis = state.analysis_results.get("technical_analysis", {})
        if technical_analysis.get("manipulation_detected", False):
            recommendations.append("Technical analysis suggests possible manipulation - verify with experts")
        
        # EXIF recommendations
        content_analysis = state.analysis_results.get("content_analysis", {})
        exif_analysis = content_analysis.get("exif_analysis", {})
        if not exif_analysis.get("has_exif", False):
            recommendations.append("Missing EXIF data - image may have been processed or edited")
        
        # Reverse search recommendations
        reverse_search = state.analysis_results.get("reverse_search", {})
        if reverse_search.get("sources_found", 0) == 0:
            recommendations.append("No sources found - image may be new, rare, or potentially fabricated")
        elif reverse_search.get("sources_found", 0) > 5:
            recommendations.append("Image widely available - check context and original source")
        
        # Source verification recommendations
        source_verification = state.analysis_results.get("source_verification", {})
        source_correlation = source_verification.get("source_correlation", {})
        if source_correlation.get("social_media_sources", 0) > source_correlation.get("news_sources", 0):
            recommendations.append("Primarily found on social media - verify through official news sources")
        
        return recommendations
    
    async def execute(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the image analysis workflow."""
        from graph.graph_manager import GraphState
        
        initial_state = GraphState(
            content_type="image",
            content=content,
            metadata=metadata or {},
            analysis_results={},
            error=None,
            processing_stage="initial",
            workflow_config=config or {}
        )
        
        # Execute workflow
        thread_config = {"configurable": {"thread_id": f"image_{asyncio.current_task().get_name()}"}}
        final_state = await self.workflow.ainvoke(initial_state, config=thread_config)
        
        return final_state