"""
Report generator node for creating detailed analysis explanations.

This module implements the report generator node that creates comprehensive,
user-friendly reports explaining the analysis results and recommendations.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from graph.nodes.content_analyzer import ContentAnalysisState, ContentType
from graph.nodes.risk_assessor import RiskLevel, RiskAssessment
from utils.logger import get_logger
from utils.exceptions import ContentProcessingError

logger = get_logger(__name__)


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    report_id: str
    content_type: str
    risk_level: str
    risk_score: float
    confidence: float
    executive_summary: str
    detailed_findings: List[Dict[str, Any]]
    recommendations: List[str]
    educational_suggestions: List[str]
    technical_details: Dict[str, Any]
    sources_and_references: List[str]
    timestamp: str
    processing_metadata: Dict[str, Any]


class ReportGeneratorNode:
    """
    Report generator node for creating detailed analysis explanations.
    
    This node creates comprehensive, user-friendly reports that explain
    the analysis results in clear, actionable terms for end users.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ReportGeneratorNode")
    
    async def generate_report(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """
        Generate comprehensive analysis report.
        
        Args:
            state: Current workflow state with analysis and risk assessment results
            
        Returns:
            ContentAnalysisState: Updated state with generated report
        """
        try:
            self.logger.info(f"Starting report generation for {state.content_type.value} content")
            state.processing_stage = "report_generation"
            
            # Extract analysis results
            content_analysis = state.analysis_results.get("content_analysis", {})
            risk_assessment_data = state.analysis_results.get("risk_assessment", {})
            
            # Create risk assessment object
            risk_assessment = self._create_risk_assessment_from_dict(risk_assessment_data)
            
            # Generate content-specific report
            if state.content_type == ContentType.TEXT:
                report = await self._generate_text_report(state, content_analysis, risk_assessment)
            elif state.content_type == ContentType.IMAGE:
                report = await self._generate_image_report(state, content_analysis, risk_assessment)
            elif state.content_type == ContentType.URL:
                report = await self._generate_url_report(state, content_analysis, risk_assessment)
            else:
                raise ContentProcessingError(f"Unsupported content type: {state.content_type}")
            
            # Update state with generated report
            state.analysis_results["final_report"] = report.__dict__
            state.processing_stage = "report_generation_complete"
            
            self.logger.info(f"Report generation completed for {state.content_type.value}")
            return state
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            state.error = f"Report generation failed: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _create_risk_assessment_from_dict(self, data: Dict[str, Any]) -> RiskAssessment:
        """Create RiskAssessment object from dictionary data."""
        return RiskAssessment(
            risk_score=data.get("risk_score", 0.5),
            risk_level=RiskLevel(data.get("risk_level", "Medium")),
            confidence=data.get("confidence", 0.5),
            contributing_factors=data.get("contributing_factors", []),
            risk_breakdown=data.get("risk_breakdown", {}),
            recommendations=data.get("recommendations", [])
        )
    
    async def _generate_text_report(
        self, 
        state: ContentAnalysisState, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> AnalysisReport:
        """Generate report for text content analysis."""
        
        # Generate executive summary
        executive_summary = self._generate_text_executive_summary(content_analysis, risk_assessment)
        
        # Generate detailed findings
        detailed_findings = self._generate_text_detailed_findings(content_analysis, risk_assessment)
        
        # Generate educational suggestions
        educational_suggestions = self._generate_text_educational_suggestions(
            content_analysis, risk_assessment
        )
        
        # Compile technical details
        technical_details = self._compile_text_technical_details(content_analysis)
        
        # Generate sources and references
        sources_and_references = self._generate_text_sources_and_references(content_analysis)
        
        return AnalysisReport(
            report_id=f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            content_type="text",
            risk_level=risk_assessment.risk_level.value,
            risk_score=risk_assessment.risk_score,
            confidence=risk_assessment.confidence,
            executive_summary=executive_summary,
            detailed_findings=detailed_findings,
            recommendations=risk_assessment.recommendations,
            educational_suggestions=educational_suggestions,
            technical_details=technical_details,
            sources_and_references=sources_and_references,
            timestamp=datetime.now().isoformat(),
            processing_metadata=self._generate_processing_metadata(state)
        )
    
    async def _generate_image_report(
        self, 
        state: ContentAnalysisState, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> AnalysisReport:
        """Generate report for image content analysis."""
        
        # Generate executive summary
        executive_summary = self._generate_image_executive_summary(content_analysis, risk_assessment)
        
        # Generate detailed findings
        detailed_findings = self._generate_image_detailed_findings(content_analysis, risk_assessment)
        
        # Generate educational suggestions
        educational_suggestions = self._generate_image_educational_suggestions(
            content_analysis, risk_assessment
        )
        
        # Compile technical details
        technical_details = self._compile_image_technical_details(content_analysis)
        
        # Generate sources and references
        sources_and_references = self._generate_image_sources_and_references(content_analysis)
        
        return AnalysisReport(
            report_id=f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            content_type="image",
            risk_level=risk_assessment.risk_level.value,
            risk_score=risk_assessment.risk_score,
            confidence=risk_assessment.confidence,
            executive_summary=executive_summary,
            detailed_findings=detailed_findings,
            recommendations=risk_assessment.recommendations,
            educational_suggestions=educational_suggestions,
            technical_details=technical_details,
            sources_and_references=sources_and_references,
            timestamp=datetime.now().isoformat(),
            processing_metadata=self._generate_processing_metadata(state)
        )
    
    async def _generate_url_report(
        self, 
        state: ContentAnalysisState, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> AnalysisReport:
        """Generate report for URL content analysis."""
        
        # Generate executive summary
        executive_summary = self._generate_url_executive_summary(content_analysis, risk_assessment)
        
        # Generate detailed findings
        detailed_findings = self._generate_url_detailed_findings(content_analysis, risk_assessment)
        
        # Generate educational suggestions
        educational_suggestions = self._generate_url_educational_suggestions(
            content_analysis, risk_assessment
        )
        
        # Compile technical details
        technical_details = self._compile_url_technical_details(content_analysis)
        
        # Generate sources and references
        sources_and_references = self._generate_url_sources_and_references(content_analysis)
        
        return AnalysisReport(
            report_id=f"url_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            content_type="url",
            risk_level=risk_assessment.risk_level.value,
            risk_score=risk_assessment.risk_score,
            confidence=risk_assessment.confidence,
            executive_summary=executive_summary,
            detailed_findings=detailed_findings,
            recommendations=risk_assessment.recommendations,
            educational_suggestions=educational_suggestions,
            technical_details=technical_details,
            sources_and_references=sources_and_references,
            timestamp=datetime.now().isoformat(),
            processing_metadata=self._generate_processing_metadata(state)
        )
    
    def _generate_text_executive_summary(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> str:
        """Generate executive summary for text analysis."""
        risk_level = risk_assessment.risk_level.value
        risk_score = risk_assessment.risk_score
        confidence = risk_assessment.confidence
        
        # Get key metrics
        text_metrics = content_analysis.get("text_metrics", {})
        word_count = text_metrics.get("word_count", 0)
        indicators_count = len(content_analysis.get("initial_indicators", []))
        
        summary_parts = [
            f"This text content has been assessed as **{risk_level} Risk** "
            f"(score: {risk_score:.2f}/1.0) with {confidence:.0%} confidence.",
            f"The analysis examined {word_count} words and identified {indicators_count} "
            f"potential misinformation indicators."
        ]
        
        # Add risk-specific messaging
        if risk_level == "High":
            summary_parts.append(
                "Multiple concerning patterns were detected that suggest this content "
                "may contain misinformation or manipulative elements."
            )
        elif risk_level == "Medium":
            summary_parts.append(
                "Some concerning patterns were detected that warrant further verification "
                "before accepting the information as reliable."
            )
        else:
            summary_parts.append(
                "The content appears relatively reliable, though standard verification "
                "practices are still recommended."
            )
        
        # Add top contributing factor
        if risk_assessment.contributing_factors:
            top_factor = risk_assessment.contributing_factors[0]
            factor_name = top_factor["factor"].replace("_", " ").title()
            summary_parts.append(
                f"The primary concern identified was {factor_name.lower()} "
                f"(impact: {top_factor['impact']:.2f})."
            )
        
        return " ".join(summary_parts)
    
    def _generate_image_executive_summary(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> str:
        """Generate executive summary for image analysis."""
        risk_level = risk_assessment.risk_level.value
        risk_score = risk_assessment.risk_score
        confidence = risk_assessment.confidence
        
        manipulation_detected = content_analysis.get("manipulation_detected", False)
        manipulation_types = content_analysis.get("manipulation_types", [])
        
        summary_parts = [
            f"This image has been assessed as **{risk_level} Risk** "
            f"(score: {risk_score:.2f}/1.0) with {confidence:.0%} confidence."
        ]
        
        if manipulation_detected:
            summary_parts.append(
                f"Digital manipulation was detected with {len(manipulation_types)} "
                f"specific manipulation types identified."
            )
        else:
            summary_parts.append("No clear digital manipulation was detected in the technical analysis.")
        
        # Add risk-specific messaging
        if risk_level == "High":
            summary_parts.append(
                "Significant concerns were identified that suggest this image may be "
                "manipulated or misleading."
            )
        elif risk_level == "Medium":
            summary_parts.append(
                "Some concerns were identified that warrant verification of the image's "
                "authenticity and context."
            )
        else:
            summary_parts.append(
                "The image appears technically sound, though context verification "
                "is still recommended."
            )
        
        return " ".join(summary_parts)
    
    def _generate_url_executive_summary(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> str:
        """Generate executive summary for URL analysis."""
        risk_level = risk_assessment.risk_level.value
        risk_score = risk_assessment.risk_score
        confidence = risk_assessment.confidence
        
        source_credibility = content_analysis.get("source_credibility", 0.5)
        domain_reputation = content_analysis.get("domain_reputation", "unknown")
        
        summary_parts = [
            f"This source has been assessed as **{risk_level} Risk** "
            f"(score: {risk_score:.2f}/1.0) with {confidence:.0%} confidence.",
            f"Source credibility score: {source_credibility:.2f}/1.0, "
            f"Domain reputation: {domain_reputation}."
        ]
        
        # Add risk-specific messaging
        if risk_level == "High":
            summary_parts.append(
                "Significant credibility concerns were identified with this source. "
                "Information should be verified through multiple independent sources."
            )
        elif risk_level == "Medium":
            summary_parts.append(
                "Some credibility concerns were identified. Cross-referencing with "
                "established sources is recommended."
            )
        else:
            summary_parts.append(
                "The source appears relatively credible, though standard verification "
                "practices remain advisable."
            )
        
        return " ".join(summary_parts)
    
    def _generate_text_detailed_findings(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings for text analysis."""
        findings = []
        
        # Content quality findings
        quality_score = content_analysis.get("quality_score", 0.5)
        text_metrics = content_analysis.get("text_metrics", {})
        
        findings.append({
            "category": "Content Quality",
            "severity": "low" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "high",
            "title": f"Content Quality Score: {quality_score:.2f}",
            "description": f"Based on text structure, length ({text_metrics.get('word_count', 0)} words), "
                          f"and formatting patterns.",
            "details": {
                "word_count": text_metrics.get("word_count", 0),
                "sentence_count": text_metrics.get("sentence_count", 0),
                "caps_ratio": text_metrics.get("all_caps_ratio", 0),
                "has_urls": text_metrics.get("has_urls", False)
            }
        })
        
        # Misinformation indicators
        indicators = content_analysis.get("initial_indicators", [])
        if indicators:
            for indicator in indicators[:5]:  # Top 5 indicators
                severity_level = "high" if indicator.get("severity", 0) > 0.7 else \
                               "medium" if indicator.get("severity", 0) > 0.4 else "low"
                
                findings.append({
                    "category": "Misinformation Indicators",
                    "severity": severity_level,
                    "title": indicator.get("type", "Unknown").replace("_", " ").title(),
                    "description": indicator.get("description", ""),
                    "details": {
                        "evidence": indicator.get("evidence", []),
                        "explanation": indicator.get("explanation", ""),
                        "severity_score": indicator.get("severity", 0)
                    }
                })
        
        # Risk factor breakdown
        for factor in risk_assessment.contributing_factors[:3]:  # Top 3 factors
            severity_level = "high" if factor["impact"] > 0.3 else \
                           "medium" if factor["impact"] > 0.15 else "low"
            
            findings.append({
                "category": "Risk Factors",
                "severity": severity_level,
                "title": factor["factor"].replace("_", " ").title(),
                "description": factor["description"],
                "details": {
                    "risk_value": factor["risk_value"],
                    "weight": factor["weight"],
                    "impact": factor["impact"]
                }
            })
        
        return findings
    
    def _generate_image_detailed_findings(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings for image analysis."""
        findings = []
        
        # Technical analysis findings
        manipulation_detected = content_analysis.get("manipulation_detected", False)
        technical_indicators = content_analysis.get("technical_indicators", [])
        
        if manipulation_detected or technical_indicators:
            findings.append({
                "category": "Technical Analysis",
                "severity": "high" if manipulation_detected else "medium",
                "title": "Digital Manipulation Detection",
                "description": "Technical analysis of the image for signs of digital manipulation.",
                "details": {
                    "manipulation_detected": manipulation_detected,
                    "manipulation_types": content_analysis.get("manipulation_types", []),
                    "technical_indicators": technical_indicators
                }
            })
        
        # Image quality findings
        image_metrics = content_analysis.get("image_metrics", {})
        quality_score = content_analysis.get("quality_score", 0.5)
        
        findings.append({
            "category": "Image Quality",
            "severity": "low" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "high",
            "title": f"Image Quality Assessment: {quality_score:.2f}",
            "description": "Analysis of image technical properties and quality indicators.",
            "details": {
                "file_size": image_metrics.get("file_size", 0),
                "dimensions": image_metrics.get("dimensions", {}),
                "format": image_metrics.get("format", "unknown"),
                "has_exif": image_metrics.get("has_exif", False)
            }
        })
        
        # Risk factor breakdown
        for factor in risk_assessment.contributing_factors[:3]:
            severity_level = "high" if factor["impact"] > 0.3 else \
                           "medium" if factor["impact"] > 0.15 else "low"
            
            findings.append({
                "category": "Risk Factors",
                "severity": severity_level,
                "title": factor["factor"].replace("_", " ").title(),
                "description": factor["description"],
                "details": {
                    "risk_value": factor["risk_value"],
                    "weight": factor["weight"],
                    "impact": factor["impact"]
                }
            })
        
        return findings
    
    def _generate_url_detailed_findings(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> List[Dict[str, Any]]:
        """Generate detailed findings for URL analysis."""
        findings = []
        
        # Source credibility findings
        source_credibility = content_analysis.get("source_credibility", 0.5)
        domain_reputation = content_analysis.get("domain_reputation", "unknown")
        
        findings.append({
            "category": "Source Credibility",
            "severity": "high" if source_credibility < 0.4 else \
                       "medium" if source_credibility < 0.7 else "low",
            "title": f"Source Credibility: {source_credibility:.2f}",
            "description": f"Domain reputation: {domain_reputation}",
            "details": {
                "credibility_score": source_credibility,
                "domain_reputation": domain_reputation,
                "bias_rating": content_analysis.get("bias_rating", "unknown")
            }
        })
        
        # URL metrics findings
        url_metrics = content_analysis.get("url_metrics", {})
        
        findings.append({
            "category": "URL Analysis",
            "severity": "low" if url_metrics.get("is_https") else "medium",
            "title": "URL Structure and Security",
            "description": "Analysis of URL structure, security, and domain properties.",
            "details": {
                "domain": url_metrics.get("domain", ""),
                "is_https": url_metrics.get("is_https", False),
                "domain_age": url_metrics.get("domain_age", 0),
                "has_redirect": url_metrics.get("has_redirect", False)
            }
        })
        
        # Content indicators
        content_indicators = content_analysis.get("content_indicators", [])
        if content_indicators:
            for indicator in content_indicators[:3]:
                severity_level = "high" if indicator.get("severity", 0) > 0.7 else \
                               "medium" if indicator.get("severity", 0) > 0.4 else "low"
                
                findings.append({
                    "category": "Content Analysis",
                    "severity": severity_level,
                    "title": indicator.get("type", "Unknown").replace("_", " ").title(),
                    "description": indicator.get("description", ""),
                    "details": {
                        "evidence": indicator.get("evidence", []),
                        "explanation": indicator.get("explanation", ""),
                        "severity_score": indicator.get("severity", 0)
                    }
                })
        
        return findings
    
    def _generate_text_educational_suggestions(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> List[str]:
        """Generate educational suggestions for text analysis."""
        suggestions = []
        
        # Base suggestions
        suggestions.extend([
            "Learn to identify emotional manipulation in text",
            "Practice fact-checking techniques for claims and statistics",
            "Understand common logical fallacies used in misinformation"
        ])
        
        # Risk-specific suggestions
        if risk_assessment.risk_level == RiskLevel.HIGH:
            suggestions.extend([
                "Study advanced misinformation detection techniques",
                "Learn about propaganda and persuasion tactics",
                "Practice lateral reading for source verification"
            ])
        
        # Factor-specific suggestions
        risk_factors = risk_assessment.risk_breakdown
        if risk_factors.get("emotional_manipulation", 0) > 0.6:
            suggestions.append("Learn to recognize emotional manipulation tactics in media")
        
        if risk_factors.get("factual_consistency", 0) > 0.6:
            suggestions.append("Practice using fact-checking websites and databases")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _generate_image_educational_suggestions(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> List[str]:
        """Generate educational suggestions for image analysis."""
        suggestions = [
            "Learn to use reverse image search tools effectively",
            "Understand basic digital image forensics techniques",
            "Practice identifying common image manipulation signs"
        ]
        
        if content_analysis.get("manipulation_detected"):
            suggestions.extend([
                "Study deepfake and AI-generated image detection",
                "Learn about metadata analysis for image verification"
            ])
        
        return suggestions[:5]
    
    def _generate_url_educational_suggestions(
        self, 
        content_analysis: Dict[str, Any], 
        risk_assessment: RiskAssessment
    ) -> List[str]:
        """Generate educational suggestions for URL analysis."""
        suggestions = [
            "Learn to evaluate source credibility and bias",
            "Practice lateral reading techniques for verification",
            "Understand media literacy and source evaluation"
        ]
        
        if content_analysis.get("source_credibility", 1) < 0.5:
            suggestions.extend([
                "Study how to research publication backgrounds",
                "Learn to identify reliable vs unreliable news sources"
            ])
        
        return suggestions[:5]
    
    def _compile_text_technical_details(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile technical details for text analysis."""
        return {
            "text_metrics": content_analysis.get("text_metrics", {}),
            "processing_time": content_analysis.get("processing_time", 0),
            "language_detected": content_analysis.get("language_detected", "unknown"),
            "requires_fact_check": content_analysis.get("requires_fact_check", False),
            "analysis_version": "1.0"
        }
    
    def _compile_image_technical_details(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile technical details for image analysis."""
        return {
            "image_metrics": content_analysis.get("image_metrics", {}),
            "processing_time": content_analysis.get("processing_time", 0),
            "requires_technical_analysis": content_analysis.get("requires_technical_analysis", False),
            "reverse_search_needed": content_analysis.get("reverse_search_needed", True),
            "analysis_version": "1.0"
        }
    
    def _compile_url_technical_details(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile technical details for URL analysis."""
        return {
            "url_metrics": content_analysis.get("url_metrics", {}),
            "processing_time": content_analysis.get("processing_time", 0),
            "requires_fact_check": content_analysis.get("requires_fact_check", False),
            "analysis_version": "1.0"
        }
    
    def _generate_text_sources_and_references(self, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate sources and references for text analysis."""
        return [
            "Misinformation detection methodology based on linguistic analysis",
            "Fact-checking best practices from journalism standards",
            "Emotional manipulation detection techniques"
        ]
    
    def _generate_image_sources_and_references(self, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate sources and references for image analysis."""
        return [
            "Digital image forensics techniques and standards",
            "Reverse image search methodology",
            "Image manipulation detection algorithms"
        ]
    
    def _generate_url_sources_and_references(self, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate sources and references for URL analysis."""
        return [
            "Source credibility evaluation frameworks",
            "Domain reputation assessment methods",
            "Media bias and reliability databases"
        ]
    
    def _generate_processing_metadata(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Generate processing metadata for the report."""
        return {
            "content_type": state.content_type.value,
            "processing_stages": [
                "content_analysis",
                "risk_assessment", 
                "report_generation"
            ],
            "analysis_id": f"{state.content_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workflow_version": "1.0",
            "error_occurred": state.error is not None,
            "final_stage": state.processing_stage
        }