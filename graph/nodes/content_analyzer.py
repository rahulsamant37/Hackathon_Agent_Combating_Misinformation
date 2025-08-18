"""
Content analyzer node for initial processing and classification.

This module implements the content analyzer node that performs initial processing
of submitted content and determines the appropriate analysis workflow.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from llm.langchain_pipeline import LangChainPipeline
from utils.logger import get_logger
from utils.exceptions import ContentProcessingError

logger = get_logger(__name__)


class ContentType(Enum):
    """Supported content types for analysis."""
    TEXT = "text"
    IMAGE = "image"
    URL = "url"


@dataclass
class ContentAnalysisState:
    """State object for content analysis workflow."""
    content_type: ContentType
    content: str
    metadata: Dict[str, Any]
    analysis_results: Dict[str, Any]
    error: Optional[str] = None
    processing_stage: str = "initial"


class ContentAnalyzerNode:
    """
    Content analyzer node for initial processing and classification.
    
    This node performs the first stage of analysis, including:
    - Content type validation and classification
    - Initial preprocessing and normalization
    - Metadata extraction and validation
    - Content quality assessment
    """
    
    def __init__(self, pipeline: LangChainPipeline):
        self.pipeline = pipeline
        self.logger = get_logger(f"{__name__}.ContentAnalyzerNode")
    
    async def analyze_content(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """
        Analyze and classify content for further processing.
        
        Args:
            state: Current workflow state containing content and metadata
            
        Returns:
            ContentAnalysisState: Updated state with analysis results
        """
        try:
            self.logger.info(f"Starting content analysis for {state.content_type.value} content")
            state.processing_stage = "content_analysis"
            
            # Perform content-specific analysis
            if state.content_type == ContentType.TEXT:
                analysis_result = await self._analyze_text_content(state)
            elif state.content_type == ContentType.IMAGE:
                analysis_result = await self._analyze_image_content(state)
            elif state.content_type == ContentType.URL:
                analysis_result = await self._analyze_url_content(state)
            else:
                raise ContentProcessingError(f"Unsupported content type: {state.content_type}")
            
            # Update state with analysis results
            state.analysis_results["content_analysis"] = analysis_result
            state.processing_stage = "content_analysis_complete"
            
            self.logger.info(f"Content analysis completed for {state.content_type.value}")
            return state
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            state.error = f"Content analysis failed: {str(e)}"
            state.processing_stage = "error"
            return state
    
    async def _analyze_text_content(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Analyze text content for initial processing."""
        content = state.content
        metadata = state.metadata
        
        # Extract basic text metrics
        text_metrics = {
            "length": len(content),
            "word_count": len(content.split()),
            "sentence_count": content.count('.') + content.count('!') + content.count('?'),
            "paragraph_count": content.count('\n\n') + 1,
            "has_urls": "http" in content.lower(),
            "has_mentions": "@" in content,
            "has_hashtags": "#" in content,
            "all_caps_ratio": sum(1 for c in content if c.isupper()) / len(content) if content else 0
        }
        
        # Perform initial LLM analysis
        try:
            llm_result = await self.pipeline.analyze_text(
                text_content=content,
                context=f"Text metrics: {text_metrics}"
            )
            
            return {
                "text_metrics": text_metrics,
                "initial_risk_score": llm_result.risk_score,
                "initial_indicators": [indicator.dict() for indicator in llm_result.indicators],
                "content_summary": llm_result.summary,
                "processing_time": metadata.get("processing_time", 0),
                "quality_score": self._calculate_text_quality_score(text_metrics),
                "requires_fact_check": self._requires_fact_checking(content, llm_result.indicators),
                "language_detected": metadata.get("language", "unknown")
            }
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed, using fallback: {str(e)}")
            return {
                "text_metrics": text_metrics,
                "initial_risk_score": 0.5,
                "initial_indicators": [],
                "content_summary": "Initial analysis failed",
                "processing_time": metadata.get("processing_time", 0),
                "quality_score": self._calculate_text_quality_score(text_metrics),
                "requires_fact_check": True,
                "language_detected": metadata.get("language", "unknown"),
                "analysis_error": str(e)
            }
    
    async def _analyze_image_content(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Analyze image content for initial processing."""
        content = state.content  # This would be image description or path
        metadata = state.metadata
        
        # Extract image metadata and metrics
        image_metrics = {
            "file_size": metadata.get("file_size", 0),
            "dimensions": metadata.get("dimensions", {}),
            "format": metadata.get("format", "unknown"),
            "has_exif": bool(metadata.get("exif_data")),
            "compression_ratio": metadata.get("compression_ratio", 0),
            "color_depth": metadata.get("color_depth", 0),
            "has_transparency": metadata.get("has_transparency", False)
        }
        
        # Perform initial LLM analysis
        try:
            llm_result = await self.pipeline.analyze_image(
                image_description=content,
                image_metadata=metadata,
                context=f"Image metrics: {image_metrics}"
            )
            
            return {
                "image_metrics": image_metrics,
                "initial_risk_score": llm_result.risk_score,
                "manipulation_detected": llm_result.manipulation_detected,
                "manipulation_types": llm_result.manipulation_types,
                "technical_indicators": llm_result.technical_indicators,
                "content_summary": llm_result.summary,
                "processing_time": metadata.get("processing_time", 0),
                "quality_score": self._calculate_image_quality_score(image_metrics),
                "requires_technical_analysis": self._requires_technical_analysis(llm_result),
                "reverse_search_needed": True
            }
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed, using fallback: {str(e)}")
            return {
                "image_metrics": image_metrics,
                "initial_risk_score": 0.5,
                "manipulation_detected": False,
                "manipulation_types": [],
                "technical_indicators": [],
                "content_summary": "Initial analysis failed",
                "processing_time": metadata.get("processing_time", 0),
                "quality_score": self._calculate_image_quality_score(image_metrics),
                "requires_technical_analysis": True,
                "reverse_search_needed": True,
                "analysis_error": str(e)
            }
    
    async def _analyze_url_content(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Analyze URL content for initial processing."""
        url = state.content
        metadata = state.metadata
        
        # Extract URL and domain metrics
        url_metrics = {
            "domain": metadata.get("domain", ""),
            "subdomain": metadata.get("subdomain", ""),
            "path_length": len(metadata.get("path", "")),
            "has_parameters": bool(metadata.get("parameters")),
            "is_https": url.startswith("https://"),
            "domain_age": metadata.get("domain_age", 0),
            "has_redirect": metadata.get("has_redirect", False),
            "content_length": metadata.get("content_length", 0)
        }
        
        # Perform initial LLM analysis
        try:
            llm_result = await self.pipeline.analyze_url(
                url=url,
                domain_info=metadata.get("domain_info", {}),
                content_summary=metadata.get("content_summary", ""),
                context=f"URL metrics: {url_metrics}"
            )
            
            return {
                "url_metrics": url_metrics,
                "initial_risk_score": llm_result.risk_score,
                "source_credibility": llm_result.source_info.credibility_score,
                "bias_rating": llm_result.source_info.bias_rating,
                "content_indicators": [indicator.dict() for indicator in llm_result.content_indicators],
                "content_summary": llm_result.summary,
                "processing_time": metadata.get("processing_time", 0),
                "quality_score": self._calculate_url_quality_score(url_metrics, llm_result.source_info),
                "requires_fact_check": self._requires_url_fact_checking(llm_result),
                "domain_reputation": llm_result.source_info.reputation
            }
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed, using fallback: {str(e)}")
            return {
                "url_metrics": url_metrics,
                "initial_risk_score": 0.5,
                "source_credibility": 0.5,
                "bias_rating": "unknown",
                "content_indicators": [],
                "content_summary": "Initial analysis failed",
                "processing_time": metadata.get("processing_time", 0),
                "quality_score": 0.5,
                "requires_fact_check": True,
                "domain_reputation": "unknown",
                "analysis_error": str(e)
            }
    
    def _calculate_text_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate quality score for text content."""
        score = 1.0
        
        # Penalize very short or very long content
        word_count = metrics.get("word_count", 0)
        if word_count < 10:
            score -= 0.3
        elif word_count > 2000:
            score -= 0.2
        
        # Penalize excessive caps
        caps_ratio = metrics.get("all_caps_ratio", 0)
        if caps_ratio > 0.3:
            score -= 0.4
        elif caps_ratio > 0.1:
            score -= 0.2
        
        # Reward proper sentence structure
        sentence_count = metrics.get("sentence_count", 0)
        if sentence_count > 0 and word_count / sentence_count > 5:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_image_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate quality score for image content."""
        score = 1.0
        
        # Check file size (too small might be low quality, too large might be suspicious)
        file_size = metrics.get("file_size", 0)
        if file_size < 10000:  # Less than 10KB
            score -= 0.3
        elif file_size > 10000000:  # More than 10MB
            score -= 0.2
        
        # Reward EXIF data presence (indicates original image)
        if metrics.get("has_exif", False):
            score += 0.1
        
        # Check dimensions
        dimensions = metrics.get("dimensions", {})
        width = dimensions.get("width", 0)
        height = dimensions.get("height", 0)
        if width > 0 and height > 0:
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_url_quality_score(self, metrics: Dict[str, Any], source_info) -> float:
        """Calculate quality score for URL content."""
        score = 1.0
        
        # Reward HTTPS
        if metrics.get("is_https", False):
            score += 0.1
        
        # Penalize very long paths (might indicate suspicious URLs)
        path_length = metrics.get("path_length", 0)
        if path_length > 100:
            score -= 0.2
        
        # Consider domain age
        domain_age = metrics.get("domain_age", 0)
        if domain_age > 365:  # More than a year old
            score += 0.1
        elif domain_age < 30:  # Less than a month old
            score -= 0.3
        
        # Use source credibility
        credibility = getattr(source_info, 'credibility_score', 0.5)
        score = (score + credibility) / 2
        
        return max(0.0, min(1.0, score))
    
    def _requires_fact_checking(self, content: str, indicators: List) -> bool:
        """Determine if content requires fact-checking."""
        # Check for factual claims
        fact_keywords = ["study shows", "research proves", "scientists say", "according to", 
                        "statistics show", "data reveals", "experts claim"]
        
        content_lower = content.lower()
        has_factual_claims = any(keyword in content_lower for keyword in fact_keywords)
        
        # Check indicators for factual inconsistencies
        has_factual_indicators = any(
            indicator.type in ["factual_claim", "statistical_claim", "scientific_claim"]
            for indicator in indicators
        )
        
        return has_factual_claims or has_factual_indicators
    
    def _requires_technical_analysis(self, llm_result) -> bool:
        """Determine if image requires technical analysis."""
        return (llm_result.manipulation_detected or 
                len(llm_result.technical_indicators) > 0 or
                llm_result.risk_score > 0.6)
    
    def _requires_url_fact_checking(self, llm_result) -> bool:
        """Determine if URL content requires fact-checking."""
        return (llm_result.risk_score > 0.4 or
                any(indicator.type in ["factual_claim", "misleading_claim"] 
                    for indicator in llm_result.content_indicators))