"""
URL analysis workflow with source verification.

This module defines the specific workflow for analyzing URL content,
including comprehensive source credibility assessment and content verification.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.nodes.content_analyzer import ContentAnalyzerNode, ContentAnalysisState, ContentType
from graph.nodes.risk_assessor import RiskAssessorNode
from graph.nodes.report_generator import ReportGeneratorNode
from graph.nodes.source_verifier import SourceVerifierNode
from llm.langchain_pipeline import LangChainPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class URLAnalysisWorkflow:
    """
    Specialized workflow for URL content analysis.
    
    Provides enhanced URL-specific processing including domain analysis,
    content extraction, source credibility assessment, and bias detection.
    """
    
    def __init__(self, pipeline: LangChainPipeline):
        self.pipeline = pipeline
        self.logger = get_logger(f"{__name__}.URLAnalysisWorkflow")
        
        # Initialize nodes
        self.content_analyzer = ContentAnalyzerNode(pipeline)
        self.risk_assessor = RiskAssessorNode()
        self.report_generator = ReportGeneratorNode()
        self.source_verifier = SourceVerifierNode()
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the URL analysis workflow graph."""
        from graph.graph_manager import GraphState
        
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyze_url", self._analyze_url_node)
        workflow.add_node("extract_content", self._extract_content_node)
        workflow.add_node("analyze_domain", self._analyze_domain_node)
        workflow.add_node("verify_sources", self._verify_sources_node)
        workflow.add_node("assess_bias", self._assess_bias_node)
        workflow.add_node("assess_risk", self._assess_risk_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_url")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "analyze_url",
            self._route_after_url_analysis,
            {
                "extract_content": "extract_content",
                "analyze_domain": "analyze_domain",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_content",
            self._route_after_content_extraction,
            {
                "analyze_domain": "analyze_domain",
                "verify_sources": "verify_sources",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_domain",
            self._route_after_domain_analysis,
            {
                "verify_sources": "verify_sources",
                "assess_bias": "assess_bias",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "verify_sources",
            self._route_after_source_verification,
            {
                "assess_bias": "assess_bias",
                "assess_risk": "assess_risk",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "assess_bias",
            self._route_after_bias_assessment,
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
    
    async def _analyze_url_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze URL structure and basic properties."""
        try:
            self.logger.info("Starting URL analysis")
            
            url = state["content"]
            
            # Parse URL and extract basic information
            url_analysis = await self._perform_url_analysis(url)
            
            # Update analysis results
            state["analysis_results"]["url_analysis"] = url_analysis
            state["processing_stage"] = "url_analysis_complete"
            
            return state
            
        except Exception as e:
            self.logger.error(f"URL analysis failed: {str(e)}")
            state["error"] = f"URL analysis failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _extract_content_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze content from the URL."""
        try:
            self.logger.info("Extracting content from URL")
            
            url = state["content"]
            
            # Simulate content extraction (in real implementation, would fetch actual content)
            content_extraction = await self._extract_url_content(url, state["metadata"])
            
            # Perform content analysis using the content analyzer
            if content_extraction.get("content_text"):
                # Convert to ContentAnalysisState for text analysis
                analysis_state = ContentAnalysisState(
                    content_type=ContentType.TEXT,  # Analyze extracted text
                    content=content_extraction["content_text"],
                    metadata=state["metadata"],
                    analysis_results=state["analysis_results"],
                    error=state["error"],
                    processing_stage=state["processing_stage"]
                )
                
                # Analyze the extracted content
                result_state = await self.content_analyzer.analyze_content(analysis_state)
                
                # Merge results
                state["analysis_results"]["content_extraction"] = content_extraction
                if "content_analysis" in result_state.analysis_results:
                    state["analysis_results"]["extracted_content_analysis"] = result_state.analysis_results["content_analysis"]
            else:
                state["analysis_results"]["content_extraction"] = content_extraction
            
            state["processing_stage"] = "content_extraction_complete"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Content extraction failed: {str(e)}")
            state["error"] = f"Content extraction failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _analyze_domain_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive domain analysis."""
        try:
            self.logger.info("Analyzing domain properties")
            
            url = state["content"]
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Perform domain analysis
            domain_analysis = await self._perform_domain_analysis(domain, url)
            
            # Update analysis results
            state["analysis_results"]["domain_analysis"] = domain_analysis
            state["processing_stage"] = "domain_analysis_complete"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Domain analysis failed: {str(e)}")
            state["error"] = f"Domain analysis failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _verify_sources_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify sources with URL-specific enhancements."""
        try:
            self.logger.info("Verifying sources for URL content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.URL,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform source verification
            result_state = await self.source_verifier.verify_sources(analysis_state)
            
            # Add URL-specific source analysis
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
    
    async def _assess_bias_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess political and ideological bias of the source."""
        try:
            self.logger.info("Assessing source bias")
            
            # Perform bias assessment
            bias_assessment = await self._perform_bias_assessment(
                state["content"],
                state["analysis_results"]
            )
            
            # Update analysis results
            state["analysis_results"]["bias_assessment"] = bias_assessment
            state["processing_stage"] = "bias_assessment_complete"
            
            return state
            
        except Exception as e:
            self.logger.error(f"Bias assessment failed: {str(e)}")
            state["error"] = f"Bias assessment failed: {str(e)}"
            state["processing_stage"] = "error"
            return state
    
    async def _assess_risk_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk with URL-specific considerations."""
        try:
            self.logger.info("Assessing risk for URL content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.URL,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Perform risk assessment
            result_state = await self.risk_assessor.assess_risk(analysis_state)
            
            # Add URL-specific risk adjustments
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
        """Generate report with URL-specific formatting."""
        try:
            self.logger.info("Generating report for URL content")
            
            # Convert to ContentAnalysisState
            analysis_state = ContentAnalysisState(
                content_type=ContentType.URL,
                content=state["content"],
                metadata=state["metadata"],
                analysis_results=state["analysis_results"],
                error=state["error"],
                processing_stage=state["processing_stage"]
            )
            
            # Generate report
            result_state = await self.report_generator.generate_report(analysis_state)
            
            # Add URL-specific report enhancements
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
        """Handle errors with URL-specific recovery."""
        self.logger.error(f"URL workflow error: {state.get('error', 'Unknown error')}")
        
        # Create minimal error report
        error_report = {
            "error_type": "url_analysis_error",
            "error_message": state.get("error", "Unknown error"),
            "url_info": {
                "url": state.get("content", ""),
                "domain": urlparse(state.get("content", "")).netloc if state.get("content") else "unknown"
            },
            "recovery_suggestions": [
                "Check if the URL is accessible and valid",
                "Verify the URL format is correct",
                "Try accessing the URL directly in a browser",
                "Check if the website is temporarily down"
            ]
        }
        
        state["analysis_results"]["error_report"] = error_report
        state["processing_stage"] = "error_handled"
        
        return state
    
    # Routing functions
    def _route_after_url_analysis(self, state: Dict[str, Any]) -> str:
        """Route after URL analysis."""
        if state.get("error"):
            return "error"
        
        url_analysis = state["analysis_results"].get("url_analysis", {})
        
        # If URL is accessible, extract content
        if url_analysis.get("accessible", False):
            return "extract_content"
        
        # Otherwise, proceed to domain analysis
        return "analyze_domain"
    
    def _route_after_content_extraction(self, state: Dict[str, Any]) -> str:
        """Route after content extraction."""
        if state.get("error"):
            return "error"
        
        content_extraction = state["analysis_results"].get("content_extraction", {})
        
        # If content was successfully extracted, verify sources
        if content_extraction.get("content_extracted", False):
            return "verify_sources"
        
        # Otherwise, proceed to domain analysis
        return "analyze_domain"
    
    def _route_after_domain_analysis(self, state: Dict[str, Any]) -> str:
        """Route after domain analysis."""
        if state.get("error"):
            return "error"
        
        domain_analysis = state["analysis_results"].get("domain_analysis", {})
        
        # If domain has credibility concerns, prioritize source verification
        if domain_analysis.get("credibility_score", 0.5) < 0.4:
            return "verify_sources"
        
        # Otherwise, proceed to bias assessment
        return "assess_bias"
    
    def _route_after_source_verification(self, state: Dict[str, Any]) -> str:
        """Route after source verification."""
        if state.get("error"):
            return "error"
        
        # Always proceed to bias assessment after source verification
        return "assess_bias"
    
    def _route_after_bias_assessment(self, state: Dict[str, Any]) -> str:
        """Route after bias assessment."""
        if state.get("error"):
            return "error"
        
        return "assess_risk"
    
    def _route_after_risk_assessment(self, state: Dict[str, Any]) -> str:
        """Route after risk assessment."""
        if state.get("error"):
            return "error"
        
        return "generate_report"
    
    # Analysis functions
    async def _perform_url_analysis(self, url: str) -> Dict[str, Any]:
        """Perform basic URL structure analysis."""
        try:
            parsed_url = urlparse(url)
            
            analysis = {
                "url": url,
                "scheme": parsed_url.scheme,
                "domain": parsed_url.netloc,
                "path": parsed_url.path,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
                "is_https": parsed_url.scheme == "https",
                "accessible": True,  # Simulated - would actually check accessibility
                "redirect_chain": [],  # Would track redirects
                "response_code": 200,  # Simulated HTTP response
                "security_indicators": {
                    "ssl_certificate": parsed_url.scheme == "https",
                    "suspicious_tld": self._check_suspicious_tld(parsed_url.netloc),
                    "url_shortener": self._is_url_shortener(parsed_url.netloc),
                    "suspicious_patterns": self._check_suspicious_patterns(url)
                }
            }
            
            return analysis
            
        except Exception as e:
            return {
                "url": url,
                "error": f"URL parsing failed: {str(e)}",
                "accessible": False
            }
    
    async def _extract_url_content(self, url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from URL (simulated)."""
        # In a real implementation, this would use libraries like requests, BeautifulSoup, etc.
        
        # Simulate content extraction
        extraction_result = {
            "content_extracted": True,
            "content_text": metadata.get("content_summary", "Sample article content about current events..."),
            "title": "Sample Article Title",
            "author": "John Doe",
            "publication_date": "2024-01-15",
            "word_count": 500,
            "images_found": 3,
            "links_found": 10,
            "social_shares": {
                "facebook": 150,
                "twitter": 75,
                "linkedin": 25
            },
            "metadata_quality": {
                "has_title": True,
                "has_author": True,
                "has_date": True,
                "has_description": True,
                "structured_data": False
            }
        }
        
        return extraction_result
    
    async def _perform_domain_analysis(self, domain: str, url: str) -> Dict[str, Any]:
        """Perform comprehensive domain analysis."""
        analysis = {
            "domain": domain,
            "domain_age": 365,  # Simulated - would use WHOIS data
            "registrar": "Unknown",
            "country": "US",
            "ip_address": "192.168.1.1",  # Simulated
            "hosting_provider": "Unknown",
            "ssl_certificate": {
                "valid": url.startswith("https://"),
                "issuer": "Unknown",
                "expiry_date": "2024-12-31"
            },
            "reputation_scores": {
                "alexa_rank": 0,  # Would use actual ranking services
                "trust_score": 0.5,
                "spam_score": 0.2
            },
            "security_flags": {
                "malware_detected": False,
                "phishing_detected": False,
                "suspicious_activity": False
            },
            "content_categories": ["news", "politics"],  # Would be determined by analysis
            "credibility_score": self._calculate_domain_credibility(domain)
        }
        
        return analysis
    
    async def _perform_bias_assessment(self, url: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess political and ideological bias."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Simulate bias assessment
        bias_assessment = {
            "political_bias": "center",  # left, center-left, center, center-right, right
            "bias_confidence": 0.6,
            "factual_reporting": "high",  # very-high, high, mostly-factual, mixed, low, very-low
            "bias_indicators": [],
            "language_analysis": {
                "emotional_language": 0.3,
                "partisan_keywords": 0.2,
                "loaded_language": 0.1
            },
            "source_transparency": {
                "funding_disclosed": False,
                "editorial_standards": True,
                "correction_policy": True,
                "author_credentials": True
            }
        }
        
        # Add domain-specific bias information
        domain_lower = domain.lower()
        
        # Simulate known bias patterns
        if any(keyword in domain_lower for keyword in ["fox", "breitbart", "dailywire"]):
            bias_assessment["political_bias"] = "right"
            bias_assessment["bias_confidence"] = 0.9
        elif any(keyword in domain_lower for keyword in ["cnn", "msnbc", "huffpost"]):
            bias_assessment["political_bias"] = "left"
            bias_assessment["bias_confidence"] = 0.9
        elif any(keyword in domain_lower for keyword in ["reuters", "ap", "bbc"]):
            bias_assessment["political_bias"] = "center"
            bias_assessment["factual_reporting"] = "very-high"
            bias_assessment["bias_confidence"] = 0.95
        
        # Analyze extracted content for bias indicators
        extracted_content = analysis_results.get("extracted_content_analysis", {})
        if extracted_content:
            sentiment_indicators = extracted_content.get("sentiment_indicators", {})
            emotional_count = sentiment_indicators.get("emotional_language", 0)
            
            if emotional_count > 5:
                bias_assessment["bias_indicators"].append("High emotional language detected")
                bias_assessment["language_analysis"]["emotional_language"] = min(1.0, emotional_count / 10)
        
        return bias_assessment
    
    # Enhancement functions
    async def _enhance_source_verification(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add URL-specific source verification enhancements."""
        source_verification = state.analysis_results.get("source_verification", {})
        
        # Add domain reputation correlation
        domain_analysis = state.analysis_results.get("domain_analysis", {})
        domain_correlation = self._correlate_domain_with_verification(
            domain_analysis, source_verification
        )
        source_verification["domain_correlation"] = domain_correlation
        
        # Add content extraction correlation
        content_extraction = state.analysis_results.get("content_extraction", {})
        content_correlation = self._correlate_content_with_verification(
            content_extraction, source_verification
        )
        source_verification["content_correlation"] = content_correlation
        
        state.analysis_results["source_verification"] = source_verification
        return state
    
    async def _enhance_risk_assessment(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add URL-specific risk assessment enhancements."""
        risk_assessment = state.analysis_results.get("risk_assessment", {})
        
        # Adjust risk based on domain analysis
        domain_analysis = state.analysis_results.get("domain_analysis", {})
        domain_credibility = domain_analysis.get("credibility_score", 0.5)
        
        current_score = risk_assessment.get("risk_score", 0.5)
        domain_risk = 1.0 - domain_credibility
        
        # Weight domain risk at 30% of total
        adjusted_score = (current_score * 0.7) + (domain_risk * 0.3)
        risk_assessment["risk_score"] = min(1.0, adjusted_score)
        
        # Adjust based on bias assessment
        bias_assessment = state.analysis_results.get("bias_assessment", {})
        factual_reporting = bias_assessment.get("factual_reporting", "mixed")
        
        if factual_reporting in ["low", "very-low"]:
            risk_assessment["risk_score"] = min(1.0, risk_assessment["risk_score"] + 0.2)
            adjustments = risk_assessment.get("url_specific_adjustments", [])
            adjustments.append("Low factual reporting score")
            risk_assessment["url_specific_adjustments"] = adjustments
        
        # Adjust based on security flags
        security_flags = domain_analysis.get("security_flags", {})
        if any(security_flags.values()):
            risk_assessment["risk_score"] = min(1.0, risk_assessment["risk_score"] + 0.3)
            adjustments = risk_assessment.get("url_specific_adjustments", [])
            adjustments.append("Security concerns detected")
            risk_assessment["url_specific_adjustments"] = adjustments
        
        state.analysis_results["risk_assessment"] = risk_assessment
        return state
    
    async def _enhance_report_generation(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """Add URL-specific report enhancements."""
        final_report = state.analysis_results.get("final_report", {})
        
        # Add URL-specific recommendations
        url_recommendations = self._generate_url_specific_recommendations(state)
        final_report["url_specific_recommendations"] = url_recommendations
        
        # Add domain summary
        domain_analysis = state.analysis_results.get("domain_analysis", {})
        final_report["domain_summary"] = {
            "domain": domain_analysis.get("domain", "unknown"),
            "credibility_score": domain_analysis.get("credibility_score", 0.5),
            "domain_age": domain_analysis.get("domain_age", 0),
            "security_status": "secure" if not any(domain_analysis.get("security_flags", {}).values()) else "concerns"
        }
        
        # Add bias summary
        bias_assessment = state.analysis_results.get("bias_assessment", {})
        final_report["bias_summary"] = {
            "political_bias": bias_assessment.get("political_bias", "unknown"),
            "factual_reporting": bias_assessment.get("factual_reporting", "unknown"),
            "bias_confidence": bias_assessment.get("bias_confidence", 0.5)
        }
        
        state.analysis_results["final_report"] = final_report
        return state
    
    # Helper functions
    def _check_suspicious_tld(self, domain: str) -> bool:
        """Check for suspicious top-level domains."""
        suspicious_tlds = [".tk", ".ml", ".ga", ".cf", ".click", ".download", ".stream"]
        return any(domain.endswith(tld) for tld in suspicious_tlds)
    
    def _is_url_shortener(self, domain: str) -> bool:
        """Check if domain is a URL shortener."""
        shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "short.link"]
        return domain in shorteners
    
    def _check_suspicious_patterns(self, url: str) -> List[str]:
        """Check for suspicious patterns in URL."""
        patterns = []
        
        if len(url) > 200:
            patterns.append("Unusually long URL")
        
        if url.count("-") > 5:
            patterns.append("Excessive hyphens in domain")
        
        if any(char in url for char in ["@", "%"]):
            patterns.append("Suspicious characters in URL")
        
        return patterns
    
    def _calculate_domain_credibility(self, domain: str) -> float:
        """Calculate domain credibility score."""
        # Simplified credibility calculation
        credibility = 0.5  # Default neutral
        
        # Known credible domains
        credible_domains = [
            "reuters.com", "ap.org", "bbc.com", "npr.org", "pbs.org",
            "wsj.com", "nytimes.com", "washingtonpost.com", "theguardian.com"
        ]
        
        # Known problematic domains
        problematic_domains = [
            "infowars.com", "naturalnews.com", "breitbart.com", "dailymail.co.uk"
        ]
        
        domain_lower = domain.lower()
        
        if any(credible in domain_lower for credible in credible_domains):
            credibility = 0.9
        elif any(problematic in domain_lower for problematic in problematic_domains):
            credibility = 0.3
        elif domain.endswith(".gov"):
            credibility = 0.95
        elif domain.endswith(".edu"):
            credibility = 0.85
        elif domain.endswith(".org"):
            credibility = 0.7
        
        return credibility
    
    def _correlate_domain_with_verification(
        self, 
        domain_analysis: Dict[str, Any], 
        source_verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate domain analysis with source verification."""
        correlation = {
            "credibility_match": False,
            "reputation_consistency": True,
            "security_alignment": True,
            "discrepancies": []
        }
        
        domain_credibility = domain_analysis.get("credibility_score", 0.5)
        verification_credibility = source_verification.get("credibility_score", 0.5)
        
        # Check if credibility scores align
        if abs(domain_credibility - verification_credibility) < 0.2:
            correlation["credibility_match"] = True
        else:
            correlation["discrepancies"].append("Credibility scores don't align")
        
        # Check security flags
        security_flags = domain_analysis.get("security_flags", {})
        if any(security_flags.values()):
            correlation["security_alignment"] = False
            correlation["discrepancies"].append("Security concerns detected")
        
        return correlation
    
    def _correlate_content_with_verification(
        self, 
        content_extraction: Dict[str, Any], 
        source_verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate content extraction with source verification."""
        correlation = {
            "content_quality_match": True,
            "metadata_consistency": True,
            "social_signals": {},
            "discrepancies": []
        }
        
        # Check metadata quality
        metadata_quality = content_extraction.get("metadata_quality", {})
        if not all(metadata_quality.values()):
            correlation["metadata_consistency"] = False
            correlation["discrepancies"].append("Incomplete metadata")
        
        # Analyze social signals
        social_shares = content_extraction.get("social_shares", {})
        total_shares = sum(social_shares.values())
        
        correlation["social_signals"] = {
            "total_shares": total_shares,
            "engagement_level": "high" if total_shares > 1000 else "medium" if total_shares > 100 else "low",
            "viral_potential": total_shares > 10000
        }
        
        return correlation
    
    def _generate_url_specific_recommendations(self, state: ContentAnalysisState) -> List[str]:
        """Generate URL-specific recommendations."""
        recommendations = []
        
        # Domain-based recommendations
        domain_analysis = state.analysis_results.get("domain_analysis", {})
        credibility_score = domain_analysis.get("credibility_score", 0.5)
        
        if credibility_score < 0.4:
            recommendations.append("Domain has low credibility - verify information through multiple sources")
        
        security_flags = domain_analysis.get("security_flags", {})
        if any(security_flags.values()):
            recommendations.append("Security concerns detected - exercise caution when visiting this site")
        
        # Bias-based recommendations
        bias_assessment = state.analysis_results.get("bias_assessment", {})
        political_bias = bias_assessment.get("political_bias", "center")
        
        if political_bias in ["left", "right"]:
            recommendations.append(f"Source shows {political_bias}-leaning bias - seek balanced perspectives")
        
        factual_reporting = bias_assessment.get("factual_reporting", "mixed")
        if factual_reporting in ["low", "very-low"]:
            recommendations.append("Source has poor factual reporting record - verify claims independently")
        
        # Content-based recommendations
        content_extraction = state.analysis_results.get("content_extraction", {})
        metadata_quality = content_extraction.get("metadata_quality", {})
        
        if not metadata_quality.get("has_author", True):
            recommendations.append("No author information - anonymous content requires extra scrutiny")
        
        if not metadata_quality.get("has_date", True):
            recommendations.append("No publication date - check if information is current")
        
        return recommendations
    
    async def execute(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the URL analysis workflow."""
        from graph.graph_manager import GraphState
        
        initial_state = GraphState(
            content_type="url",
            content=content,
            metadata=metadata or {},
            analysis_results={},
            error=None,
            processing_stage="initial",
            workflow_config=config or {}
        )
        
        # Execute workflow
        thread_config = {"configurable": {"thread_id": f"url_{asyncio.current_task().get_name()}"}}
        final_state = await self.workflow.ainvoke(initial_state, config=thread_config)
        
        return final_state