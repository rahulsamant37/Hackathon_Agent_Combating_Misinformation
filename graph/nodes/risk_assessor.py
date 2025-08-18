"""
Risk assessor node for calculating comprehensive risk scores.

This module implements the risk assessor node that calculates final risk scores
based on multiple analysis factors and confidence levels.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from graph.nodes.content_analyzer import ContentAnalysisState, ContentType
from utils.logger import get_logger
from utils.exceptions import ContentProcessingError

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    risk_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    contributing_factors: List[Dict[str, Any]]
    risk_breakdown: Dict[str, float]
    recommendations: List[str]


class RiskAssessorNode:
    """
    Risk assessor node for comprehensive risk score calculation.
    
    This node combines multiple analysis results to produce a final risk assessment
    including risk score, confidence level, and detailed explanations.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.RiskAssessorNode")
        
        # Risk factor weights for different content types
        self.text_weights = {
            "content_quality": 0.15,
            "linguistic_indicators": 0.25,
            "factual_consistency": 0.30,
            "source_credibility": 0.20,
            "emotional_manipulation": 0.10
        }
        
        self.image_weights = {
            "technical_analysis": 0.35,
            "content_quality": 0.15,
            "manipulation_indicators": 0.30,
            "source_verification": 0.20
        }
        
        self.url_weights = {
            "source_credibility": 0.40,
            "content_quality": 0.20,
            "domain_reputation": 0.25,
            "content_indicators": 0.15
        }
    
    async def assess_risk(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """
        Assess comprehensive risk based on all analysis results.
        
        Args:
            state: Current workflow state with analysis results
            
        Returns:
            ContentAnalysisState: Updated state with risk assessment
        """
        try:
            self.logger.info(f"Starting risk assessment for {state.content_type.value} content")
            state.processing_stage = "risk_assessment"
            
            # Perform content-specific risk assessment
            if state.content_type == ContentType.TEXT:
                risk_assessment = await self._assess_text_risk(state)
            elif state.content_type == ContentType.IMAGE:
                risk_assessment = await self._assess_image_risk(state)
            elif state.content_type == ContentType.URL:
                risk_assessment = await self._assess_url_risk(state)
            else:
                raise ContentProcessingError(f"Unsupported content type: {state.content_type}")
            
            # Update state with risk assessment
            state.analysis_results["risk_assessment"] = risk_assessment.__dict__
            state.processing_stage = "risk_assessment_complete"
            
            self.logger.info(f"Risk assessment completed: {risk_assessment.risk_level.value} "
                           f"(score: {risk_assessment.risk_score:.2f}, "
                           f"confidence: {risk_assessment.confidence:.2f})")
            return state
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            state.error = f"Risk assessment failed: {str(e)}"
            state.processing_stage = "error"
            return state
    
    async def _assess_text_risk(self, state: ContentAnalysisState) -> RiskAssessment:
        """Assess risk for text content."""
        analysis_results = state.analysis_results
        content_analysis = analysis_results.get("content_analysis", {})
        
        # Calculate individual risk factors
        risk_factors = {}
        
        # Content quality factor
        quality_score = content_analysis.get("quality_score", 0.5)
        risk_factors["content_quality"] = 1.0 - quality_score
        
        # Linguistic indicators factor
        initial_risk = content_analysis.get("initial_risk_score", 0.5)
        indicators = content_analysis.get("initial_indicators", [])
        linguistic_risk = self._calculate_linguistic_risk(initial_risk, indicators)
        risk_factors["linguistic_indicators"] = linguistic_risk
        
        # Factual consistency factor (would be enhanced with fact-checking results)
        factual_risk = self._calculate_factual_risk(content_analysis, indicators)
        risk_factors["factual_consistency"] = factual_risk
        
        # Source credibility factor (if available)
        source_risk = self._calculate_source_risk(state.metadata)
        risk_factors["source_credibility"] = source_risk
        
        # Emotional manipulation factor
        emotional_risk = self._calculate_emotional_risk(indicators, content_analysis)
        risk_factors["emotional_manipulation"] = emotional_risk
        
        # Calculate weighted risk score
        weighted_score = sum(
            risk_factors[factor] * self.text_weights[factor]
            for factor in self.text_weights.keys()
        )
        
        # Calculate confidence based on data availability and consistency
        confidence = self._calculate_text_confidence(content_analysis, risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(weighted_score)
        
        # Generate contributing factors
        contributing_factors = self._generate_contributing_factors(risk_factors, self.text_weights)
        
        # Generate recommendations
        recommendations = self._generate_text_recommendations(risk_factors, risk_level)
        
        return RiskAssessment(
            risk_score=weighted_score,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            risk_breakdown=risk_factors,
            recommendations=recommendations
        )
    
    async def _assess_image_risk(self, state: ContentAnalysisState) -> RiskAssessment:
        """Assess risk for image content."""
        analysis_results = state.analysis_results
        content_analysis = analysis_results.get("content_analysis", {})
        
        # Calculate individual risk factors
        risk_factors = {}
        
        # Technical analysis factor
        manipulation_detected = content_analysis.get("manipulation_detected", False)
        technical_indicators = content_analysis.get("technical_indicators", [])
        risk_factors["technical_analysis"] = self._calculate_technical_risk(
            manipulation_detected, technical_indicators
        )
        
        # Content quality factor
        quality_score = content_analysis.get("quality_score", 0.5)
        risk_factors["content_quality"] = 1.0 - quality_score
        
        # Manipulation indicators factor
        manipulation_types = content_analysis.get("manipulation_types", [])
        initial_risk = content_analysis.get("initial_risk_score", 0.5)
        risk_factors["manipulation_indicators"] = self._calculate_manipulation_risk(
            manipulation_types, initial_risk
        )
        
        # Source verification factor (reverse image search results)
        source_risk = self._calculate_image_source_risk(state.metadata)
        risk_factors["source_verification"] = source_risk
        
        # Calculate weighted risk score
        weighted_score = sum(
            risk_factors[factor] * self.image_weights[factor]
            for factor in self.image_weights.keys()
        )
        
        # Calculate confidence
        confidence = self._calculate_image_confidence(content_analysis, risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(weighted_score)
        
        # Generate contributing factors
        contributing_factors = self._generate_contributing_factors(risk_factors, self.image_weights)
        
        # Generate recommendations
        recommendations = self._generate_image_recommendations(risk_factors, risk_level)
        
        return RiskAssessment(
            risk_score=weighted_score,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            risk_breakdown=risk_factors,
            recommendations=recommendations
        )
    
    async def _assess_url_risk(self, state: ContentAnalysisState) -> RiskAssessment:
        """Assess risk for URL content."""
        analysis_results = state.analysis_results
        content_analysis = analysis_results.get("content_analysis", {})
        
        # Calculate individual risk factors
        risk_factors = {}
        
        # Source credibility factor
        source_credibility = content_analysis.get("source_credibility", 0.5)
        risk_factors["source_credibility"] = 1.0 - source_credibility
        
        # Content quality factor
        quality_score = content_analysis.get("quality_score", 0.5)
        risk_factors["content_quality"] = 1.0 - quality_score
        
        # Domain reputation factor
        domain_reputation = self._calculate_domain_reputation_risk(content_analysis)
        risk_factors["domain_reputation"] = domain_reputation
        
        # Content indicators factor
        content_indicators = content_analysis.get("content_indicators", [])
        initial_risk = content_analysis.get("initial_risk_score", 0.5)
        risk_factors["content_indicators"] = self._calculate_content_indicators_risk(
            content_indicators, initial_risk
        )
        
        # Calculate weighted risk score
        weighted_score = sum(
            risk_factors[factor] * self.url_weights[factor]
            for factor in self.url_weights.keys()
        )
        
        # Calculate confidence
        confidence = self._calculate_url_confidence(content_analysis, risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(weighted_score)
        
        # Generate contributing factors
        contributing_factors = self._generate_contributing_factors(risk_factors, self.url_weights)
        
        # Generate recommendations
        recommendations = self._generate_url_recommendations(risk_factors, risk_level)
        
        return RiskAssessment(
            risk_score=weighted_score,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            risk_breakdown=risk_factors,
            recommendations=recommendations
        )
    
    def _calculate_linguistic_risk(self, initial_risk: float, indicators: List[Dict]) -> float:
        """Calculate risk based on linguistic indicators."""
        if not indicators:
            return initial_risk
        
        # Weight indicators by severity
        weighted_risks = []
        for indicator in indicators:
            severity = indicator.get("severity", 0.5)
            indicator_type = indicator.get("type", "")
            
            # Apply type-specific weights
            if indicator_type in ["emotional_language", "sensationalism"]:
                weighted_risks.append(severity * 1.2)
            elif indicator_type in ["logical_fallacy", "misleading_claim"]:
                weighted_risks.append(severity * 1.5)
            else:
                weighted_risks.append(severity)
        
        if weighted_risks:
            return min(1.0, statistics.mean(weighted_risks))
        return initial_risk
    
    def _calculate_factual_risk(self, content_analysis: Dict, indicators: List[Dict]) -> float:
        """Calculate risk based on factual consistency."""
        base_risk = 0.3  # Default moderate risk
        
        # Check if fact-checking is required
        requires_fact_check = content_analysis.get("requires_fact_check", False)
        if requires_fact_check:
            base_risk += 0.2
        
        # Check for factual claim indicators
        factual_indicators = [
            ind for ind in indicators 
            if ind.get("type") in ["factual_claim", "statistical_claim", "scientific_claim"]
        ]
        
        if factual_indicators:
            avg_severity = statistics.mean([ind.get("severity", 0.5) for ind in factual_indicators])
            base_risk = (base_risk + avg_severity) / 2
        
        return min(1.0, base_risk)
    
    def _calculate_source_risk(self, metadata: Dict) -> float:
        """Calculate risk based on source information."""
        # Default moderate risk if no source info
        if not metadata.get("source_info"):
            return 0.5
        
        source_info = metadata["source_info"]
        credibility = source_info.get("credibility_score", 0.5)
        
        # Invert credibility to get risk
        return 1.0 - credibility
    
    def _calculate_emotional_risk(self, indicators: List[Dict], content_analysis: Dict) -> float:
        """Calculate risk based on emotional manipulation."""
        emotional_indicators = [
            ind for ind in indicators 
            if ind.get("type") in ["emotional_language", "fear_mongering", "sensationalism"]
        ]
        
        if not emotional_indicators:
            return 0.2  # Low baseline risk
        
        avg_severity = statistics.mean([ind.get("severity", 0.5) for ind in emotional_indicators])
        
        # Consider caps ratio from text metrics
        text_metrics = content_analysis.get("text_metrics", {})
        caps_ratio = text_metrics.get("all_caps_ratio", 0)
        
        # High caps usage increases emotional manipulation risk
        caps_penalty = min(0.3, caps_ratio * 2)
        
        return min(1.0, avg_severity + caps_penalty)
    
    def _calculate_technical_risk(self, manipulation_detected: bool, technical_indicators: List) -> float:
        """Calculate risk based on technical analysis."""
        if manipulation_detected:
            return 0.8  # High risk if manipulation detected
        
        if technical_indicators:
            return 0.6  # Medium-high risk if technical indicators present
        
        return 0.2  # Low baseline risk
    
    def _calculate_manipulation_risk(self, manipulation_types: List, initial_risk: float) -> float:
        """Calculate risk based on manipulation types."""
        if not manipulation_types:
            return initial_risk
        
        # Different manipulation types have different risk levels
        high_risk_types = ["deepfake", "face_swap", "synthetic_generation"]
        medium_risk_types = ["color_adjustment", "object_removal", "background_change"]
        
        high_risk_count = sum(1 for t in manipulation_types if t in high_risk_types)
        medium_risk_count = sum(1 for t in manipulation_types if t in medium_risk_types)
        
        risk_score = initial_risk
        risk_score += high_risk_count * 0.3
        risk_score += medium_risk_count * 0.15
        
        return min(1.0, risk_score)
    
    def _calculate_image_source_risk(self, metadata: Dict) -> float:
        """Calculate risk based on image source verification."""
        # Check reverse image search results
        reverse_search = metadata.get("reverse_search_results", {})
        
        if not reverse_search:
            return 0.6  # Higher risk if no reverse search available
        
        matches_found = reverse_search.get("matches_found", 0)
        oldest_match = reverse_search.get("oldest_match_date")
        
        if matches_found == 0:
            return 0.7  # High risk if no matches found (potentially new/fake)
        elif matches_found > 10:
            return 0.3  # Lower risk if widely available
        else:
            return 0.5  # Medium risk
    
    def _calculate_domain_reputation_risk(self, content_analysis: Dict) -> float:
        """Calculate risk based on domain reputation."""
        domain_reputation = content_analysis.get("domain_reputation", "unknown")
        
        reputation_scores = {
            "excellent": 0.1,
            "good": 0.2,
            "fair": 0.4,
            "poor": 0.7,
            "very_poor": 0.9,
            "unknown": 0.5
        }
        
        return reputation_scores.get(domain_reputation.lower(), 0.5)
    
    def _calculate_content_indicators_risk(self, indicators: List[Dict], initial_risk: float) -> float:
        """Calculate risk based on content indicators."""
        if not indicators:
            return initial_risk
        
        severity_scores = [ind.get("severity", 0.5) for ind in indicators]
        return statistics.mean(severity_scores) if severity_scores else initial_risk
    
    def _calculate_text_confidence(self, content_analysis: Dict, risk_factors: Dict) -> float:
        """Calculate confidence for text analysis."""
        confidence = 0.7  # Base confidence
        
        # Increase confidence if we have good quality metrics
        if content_analysis.get("quality_score", 0) > 0.7:
            confidence += 0.1
        
        # Increase confidence if we have multiple indicators
        indicators_count = len(content_analysis.get("initial_indicators", []))
        if indicators_count > 3:
            confidence += 0.1
        
        # Decrease confidence if analysis had errors
        if content_analysis.get("analysis_error"):
            confidence -= 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_image_confidence(self, content_analysis: Dict, risk_factors: Dict) -> float:
        """Calculate confidence for image analysis."""
        confidence = 0.6  # Base confidence (lower for images)
        
        # Increase confidence if we have EXIF data
        image_metrics = content_analysis.get("image_metrics", {})
        if image_metrics.get("has_exif"):
            confidence += 0.15
        
        # Increase confidence if technical analysis was performed
        if content_analysis.get("requires_technical_analysis"):
            confidence += 0.1
        
        # Decrease confidence if analysis had errors
        if content_analysis.get("analysis_error"):
            confidence -= 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_url_confidence(self, content_analysis: Dict, risk_factors: Dict) -> float:
        """Calculate confidence for URL analysis."""
        confidence = 0.8  # Base confidence (higher for URLs)
        
        # Increase confidence if we have domain information
        url_metrics = content_analysis.get("url_metrics", {})
        if url_metrics.get("domain_age", 0) > 0:
            confidence += 0.1
        
        # Increase confidence if source credibility is available
        if content_analysis.get("source_credibility", 0) != 0.5:
            confidence += 0.1
        
        # Decrease confidence if analysis had errors
        if content_analysis.get("analysis_error"):
            confidence -= 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score."""
        if risk_score >= 0.7:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_contributing_factors(self, risk_factors: Dict, weights: Dict) -> List[Dict[str, Any]]:
        """Generate list of contributing factors with their impact."""
        factors = []
        
        for factor_name, risk_value in risk_factors.items():
            weight = weights.get(factor_name, 0)
            impact = risk_value * weight
            
            factors.append({
                "factor": factor_name,
                "risk_value": risk_value,
                "weight": weight,
                "impact": impact,
                "description": self._get_factor_description(factor_name, risk_value)
            })
        
        # Sort by impact (highest first)
        factors.sort(key=lambda x: x["impact"], reverse=True)
        return factors
    
    def _get_factor_description(self, factor_name: str, risk_value: float) -> str:
        """Get human-readable description for a risk factor."""
        descriptions = {
            "content_quality": f"Content quality assessment (risk: {risk_value:.2f})",
            "linguistic_indicators": f"Language patterns and indicators (risk: {risk_value:.2f})",
            "factual_consistency": f"Factual claims and consistency (risk: {risk_value:.2f})",
            "source_credibility": f"Source reliability and credibility (risk: {risk_value:.2f})",
            "emotional_manipulation": f"Emotional manipulation tactics (risk: {risk_value:.2f})",
            "technical_analysis": f"Technical manipulation detection (risk: {risk_value:.2f})",
            "manipulation_indicators": f"Visual manipulation indicators (risk: {risk_value:.2f})",
            "source_verification": f"Image source verification (risk: {risk_value:.2f})",
            "domain_reputation": f"Domain and website reputation (risk: {risk_value:.2f})",
            "content_indicators": f"Content-based risk indicators (risk: {risk_value:.2f})"
        }
        
        return descriptions.get(factor_name, f"{factor_name} (risk: {risk_value:.2f})")
    
    def _generate_text_recommendations(self, risk_factors: Dict, risk_level: RiskLevel) -> List[str]:
        """Generate recommendations for text content."""
        recommendations = []
        
        if risk_level == RiskLevel.HIGH:
            recommendations.append("Exercise extreme caution - this content shows multiple misinformation indicators")
            recommendations.append("Verify claims through multiple independent, reputable sources")
            recommendations.append("Consider the source's credibility and potential bias")
        
        if risk_factors.get("factual_consistency", 0) > 0.6:
            recommendations.append("Fact-check specific claims made in this content")
            recommendations.append("Look for peer-reviewed sources that support or contradict the claims")
        
        if risk_factors.get("emotional_manipulation", 0) > 0.6:
            recommendations.append("Be aware of emotional manipulation tactics in the language")
            recommendations.append("Focus on factual content rather than emotional appeals")
        
        if risk_factors.get("source_credibility", 0) > 0.6:
            recommendations.append("Research the source's background and track record")
            recommendations.append("Check if the source has editorial standards and fact-checking processes")
        
        return recommendations
    
    def _generate_image_recommendations(self, risk_factors: Dict, risk_level: RiskLevel) -> List[str]:
        """Generate recommendations for image content."""
        recommendations = []
        
        if risk_level == RiskLevel.HIGH:
            recommendations.append("This image shows signs of potential manipulation - verify authenticity")
            recommendations.append("Use reverse image search to find the original source")
        
        if risk_factors.get("technical_analysis", 0) > 0.7:
            recommendations.append("Technical analysis indicates possible digital manipulation")
            recommendations.append("Consult image forensics experts for detailed verification")
        
        if risk_factors.get("source_verification", 0) > 0.6:
            recommendations.append("Unable to verify the original source of this image")
            recommendations.append("Search for the image on fact-checking websites")
        
        return recommendations
    
    def _generate_url_recommendations(self, risk_factors: Dict, risk_level: RiskLevel) -> List[str]:
        """Generate recommendations for URL content."""
        recommendations = []
        
        if risk_level == RiskLevel.HIGH:
            recommendations.append("This source has credibility concerns - verify information elsewhere")
            recommendations.append("Cross-reference claims with established, reputable news sources")
        
        if risk_factors.get("source_credibility", 0) > 0.6:
            recommendations.append("Research the publication's editorial standards and bias")
            recommendations.append("Check the source's track record on fact-checking websites")
        
        if risk_factors.get("domain_reputation", 0) > 0.6:
            recommendations.append("This domain has reputation issues - proceed with caution")
            recommendations.append("Verify the website's legitimacy and ownership")
        
        return recommendations