"""
Source verification node for credibility checking and validation.

This module implements the source verification node that performs detailed
source credibility assessment and cross-referencing for fact-checking.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse

from graph.nodes.content_analyzer import ContentAnalysisState, ContentType
from utils.logger import get_logger
from utils.exceptions import ContentProcessingError

logger = get_logger(__name__)


@dataclass
class SourceVerificationResult:
    """Source verification result."""
    source_type: str
    credibility_score: float
    verification_status: str
    bias_assessment: str
    reputation_indicators: List[Dict[str, Any]]
    fact_check_results: List[Dict[str, Any]]
    cross_references: List[str]
    verification_confidence: float


class SourceVerifierNode:
    """
    Source verification node for credibility checking and validation.
    
    This node performs comprehensive source verification including:
    - Domain reputation assessment
    - Bias and credibility evaluation
    - Cross-referencing with known sources
    - Fact-checking database lookups
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.SourceVerifierNode")
        
        # Known credible sources (simplified for demo)
        self.credible_sources = {
            "reuters.com": {"credibility": 0.95, "bias": "center", "type": "news"},
            "ap.org": {"credibility": 0.95, "bias": "center", "type": "news"},
            "bbc.com": {"credibility": 0.90, "bias": "center-left", "type": "news"},
            "npr.org": {"credibility": 0.90, "bias": "center-left", "type": "news"},
            "wsj.com": {"credibility": 0.85, "bias": "center-right", "type": "news"},
            "nytimes.com": {"credibility": 0.85, "bias": "center-left", "type": "news"},
            "nature.com": {"credibility": 0.98, "bias": "center", "type": "scientific"},
            "science.org": {"credibility": 0.98, "bias": "center", "type": "scientific"},
            "nejm.org": {"credibility": 0.98, "bias": "center", "type": "medical"},
            "who.int": {"credibility": 0.95, "bias": "center", "type": "health"},
            "cdc.gov": {"credibility": 0.95, "bias": "center", "type": "health"}
        }
        
        # Known problematic sources (simplified for demo)
        self.problematic_sources = {
            "infowars.com": {"credibility": 0.15, "bias": "extreme-right", "type": "conspiracy"},
            "naturalnews.com": {"credibility": 0.20, "bias": "right", "type": "pseudoscience"},
            "breitbart.com": {"credibility": 0.35, "bias": "right", "type": "partisan"},
            "dailymail.co.uk": {"credibility": 0.45, "bias": "right", "type": "tabloid"}
        }
        
        # Fact-checking organizations
        self.fact_checkers = {
            "snopes.com": {"credibility": 0.90, "specialization": "general"},
            "factcheck.org": {"credibility": 0.95, "specialization": "politics"},
            "politifact.com": {"credibility": 0.90, "specialization": "politics"},
            "fullfact.org": {"credibility": 0.90, "specialization": "uk_politics"},
            "checkyourfact.com": {"credibility": 0.85, "specialization": "general"}
        }
    
    async def verify_sources(self, state: ContentAnalysisState) -> ContentAnalysisState:
        """
        Perform comprehensive source verification.
        
        Args:
            state: Current workflow state with content and analysis results
            
        Returns:
            ContentAnalysisState: Updated state with source verification results
        """
        try:
            self.logger.info(f"Starting source verification for {state.content_type.value} content")
            state.processing_stage = "source_verification"
            
            # Perform content-specific source verification
            if state.content_type == ContentType.TEXT:
                verification_result = await self._verify_text_sources(state)
            elif state.content_type == ContentType.IMAGE:
                verification_result = await self._verify_image_sources(state)
            elif state.content_type == ContentType.URL:
                verification_result = await self._verify_url_sources(state)
            else:
                raise ContentProcessingError(f"Unsupported content type: {state.content_type}")
            
            # Update state with verification results
            state.analysis_results["source_verification"] = verification_result.__dict__
            state.processing_stage = "source_verification_complete"
            
            self.logger.info(f"Source verification completed: {verification_result.verification_status}")
            return state
            
        except Exception as e:
            self.logger.error(f"Source verification failed: {str(e)}")
            state.error = f"Source verification failed: {str(e)}"
            state.processing_stage = "error"
            return state
    
    async def _verify_text_sources(self, state: ContentAnalysisState) -> SourceVerificationResult:
        """Verify sources mentioned in text content."""
        content = state.content
        content_analysis = state.analysis_results.get("content_analysis", {})
        
        # Extract URLs from text
        urls = self._extract_urls_from_text(content)
        
        # Analyze mentioned sources
        source_assessments = []
        overall_credibility = 0.5  # Default neutral
        
        if urls:
            for url in urls:
                assessment = await self._assess_url_credibility(url)
                source_assessments.append(assessment)
            
            # Calculate overall credibility
            credibility_scores = [a["credibility"] for a in source_assessments]
            overall_credibility = sum(credibility_scores) / len(credibility_scores)
        
        # Check for fact-checking opportunities
        fact_check_results = await self._check_for_existing_fact_checks(content, urls)
        
        # Generate cross-references
        cross_references = await self._generate_cross_references(content, content_analysis)
        
        # Determine verification status
        verification_status = self._determine_verification_status(
            overall_credibility, source_assessments, fact_check_results
        )
        
        # Calculate verification confidence
        verification_confidence = self._calculate_verification_confidence(
            len(urls), len(fact_check_results), len(cross_references)
        )
        
        return SourceVerificationResult(
            source_type="text_embedded",
            credibility_score=overall_credibility,
            verification_status=verification_status,
            bias_assessment=self._assess_overall_bias(source_assessments),
            reputation_indicators=source_assessments,
            fact_check_results=fact_check_results,
            cross_references=cross_references,
            verification_confidence=verification_confidence
        )
    
    async def _verify_image_sources(self, state: ContentAnalysisState) -> SourceVerificationResult:
        """Verify sources for image content."""
        metadata = state.metadata
        content_analysis = state.analysis_results.get("content_analysis", {})
        
        # Check reverse image search results
        reverse_search = metadata.get("reverse_search_results", {})
        source_urls = reverse_search.get("source_urls", [])
        
        # Assess source credibility
        source_assessments = []
        overall_credibility = 0.5
        
        if source_urls:
            for url in source_urls:
                assessment = await self._assess_url_credibility(url)
                source_assessments.append(assessment)
            
            credibility_scores = [a["credibility"] for a in source_assessments]
            overall_credibility = sum(credibility_scores) / len(credibility_scores)
        else:
            # No sources found - potentially concerning
            overall_credibility = 0.3
        
        # Check for fact-checking of the image
        fact_check_results = await self._check_image_fact_checks(metadata)
        
        # Generate cross-references based on image content
        cross_references = await self._generate_image_cross_references(
            state.content, content_analysis
        )
        
        # Determine verification status
        verification_status = self._determine_image_verification_status(
            overall_credibility, len(source_urls), fact_check_results
        )
        
        # Calculate verification confidence
        verification_confidence = self._calculate_image_verification_confidence(
            reverse_search, source_assessments, fact_check_results
        )
        
        return SourceVerificationResult(
            source_type="image_reverse_search",
            credibility_score=overall_credibility,
            verification_status=verification_status,
            bias_assessment=self._assess_overall_bias(source_assessments),
            reputation_indicators=source_assessments,
            fact_check_results=fact_check_results,
            cross_references=cross_references,
            verification_confidence=verification_confidence
        )
    
    async def _verify_url_sources(self, state: ContentAnalysisState) -> SourceVerificationResult:
        """Verify the URL source itself."""
        url = state.content
        content_analysis = state.analysis_results.get("content_analysis", {})
        
        # Assess the primary URL
        primary_assessment = await self._assess_url_credibility(url)
        
        # Check for related sources in content
        content_summary = state.metadata.get("content_summary", "")
        related_urls = self._extract_urls_from_text(content_summary)
        
        related_assessments = []
        for related_url in related_urls:
            assessment = await self._assess_url_credibility(related_url)
            related_assessments.append(assessment)
        
        # Combine assessments
        all_assessments = [primary_assessment] + related_assessments
        
        # Calculate overall credibility
        credibility_scores = [a["credibility"] for a in all_assessments]
        overall_credibility = sum(credibility_scores) / len(credibility_scores)
        
        # Check for fact-checking of the URL/content
        fact_check_results = await self._check_url_fact_checks(url, content_summary)
        
        # Generate cross-references
        cross_references = await self._generate_url_cross_references(url, content_analysis)
        
        # Determine verification status
        verification_status = self._determine_url_verification_status(
            primary_assessment, fact_check_results
        )
        
        # Calculate verification confidence
        verification_confidence = self._calculate_url_verification_confidence(
            primary_assessment, related_assessments, fact_check_results
        )
        
        return SourceVerificationResult(
            source_type="primary_url",
            credibility_score=overall_credibility,
            verification_status=verification_status,
            bias_assessment=primary_assessment.get("bias", "unknown"),
            reputation_indicators=all_assessments,
            fact_check_results=fact_check_results,
            cross_references=cross_references,
            verification_confidence=verification_confidence
        )
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text content."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return list(set(urls))  # Remove duplicates
    
    async def _assess_url_credibility(self, url: str) -> Dict[str, Any]:
        """Assess credibility of a single URL."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check against known sources
            if domain in self.credible_sources:
                source_info = self.credible_sources[domain]
                return {
                    "url": url,
                    "domain": domain,
                    "credibility": source_info["credibility"],
                    "bias": source_info["bias"],
                    "type": source_info["type"],
                    "status": "verified_credible",
                    "confidence": 0.9
                }
            
            elif domain in self.problematic_sources:
                source_info = self.problematic_sources[domain]
                return {
                    "url": url,
                    "domain": domain,
                    "credibility": source_info["credibility"],
                    "bias": source_info["bias"],
                    "type": source_info["type"],
                    "status": "verified_problematic",
                    "confidence": 0.9
                }
            
            else:
                # Unknown domain - assess based on heuristics
                credibility_score = self._assess_unknown_domain(domain, url)
                return {
                    "url": url,
                    "domain": domain,
                    "credibility": credibility_score,
                    "bias": "unknown",
                    "type": "unknown",
                    "status": "unverified",
                    "confidence": 0.4
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to assess URL {url}: {str(e)}")
            return {
                "url": url,
                "domain": "unknown",
                "credibility": 0.3,
                "bias": "unknown",
                "type": "unknown",
                "status": "assessment_failed",
                "confidence": 0.1
            }
    
    def _assess_unknown_domain(self, domain: str, url: str) -> float:
        """Assess credibility of unknown domain using heuristics."""
        score = 0.5  # Start with neutral
        
        # Government domains are generally more credible
        if domain.endswith('.gov'):
            score += 0.3
        elif domain.endswith('.edu'):
            score += 0.2
        elif domain.endswith('.org'):
            score += 0.1
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'news', 'truth', 'real', 'patriot', 'freedom', 'liberty',
            'insider', 'expose', 'reveal', 'secret', 'hidden'
        ]
        
        domain_lower = domain.lower()
        suspicious_count = sum(1 for pattern in suspicious_patterns if pattern in domain_lower)
        score -= suspicious_count * 0.1
        
        # Very new or very long domains might be suspicious
        if len(domain) > 30:
            score -= 0.1
        
        # Multiple hyphens might indicate suspicious domain
        if domain.count('-') > 2:
            score -= 0.1
        
        return max(0.1, min(0.9, score))
    
    async def _check_for_existing_fact_checks(self, content: str, urls: List[str]) -> List[Dict[str, Any]]:
        """Check for existing fact-checks of the content or sources."""
        fact_checks = []
        
        # Simulate fact-checking database lookup
        # In a real implementation, this would query actual fact-checking APIs
        
        # Check for common misinformation keywords
        misinformation_keywords = [
            "vaccine", "covid", "election", "climate change", "5g",
            "conspiracy", "deep state", "fake news", "hoax"
        ]
        
        content_lower = content.lower()
        for keyword in misinformation_keywords:
            if keyword in content_lower:
                # Simulate finding a related fact-check
                fact_checks.append({
                    "source": "factcheck.org",
                    "topic": keyword,
                    "verdict": "needs_verification",
                    "confidence": 0.6,
                    "url": f"https://factcheck.org/search/{keyword.replace(' ', '-')}",
                    "summary": f"Related fact-checks available for {keyword}"
                })
        
        return fact_checks[:3]  # Limit to 3 results
    
    async def _check_image_fact_checks(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for fact-checks related to the image."""
        fact_checks = []
        
        # Check reverse image search results for fact-checking sites
        reverse_search = metadata.get("reverse_search_results", {})
        source_urls = reverse_search.get("source_urls", [])
        
        for url in source_urls:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            if any(fact_checker in domain for fact_checker in self.fact_checkers.keys()):
                fact_checks.append({
                    "source": domain,
                    "topic": "image_verification",
                    "verdict": "fact_checked",
                    "confidence": 0.8,
                    "url": url,
                    "summary": f"Image appears in fact-checking database at {domain}"
                })
        
        return fact_checks
    
    async def _check_url_fact_checks(self, url: str, content_summary: str) -> List[Dict[str, Any]]:
        """Check for fact-checks of the URL or its content."""
        fact_checks = []
        
        # Check if the URL itself has been fact-checked
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if domain in self.problematic_sources:
            fact_checks.append({
                "source": "media_bias_fact_check",
                "topic": "source_reliability",
                "verdict": "low_credibility",
                "confidence": 0.9,
                "url": f"https://mediabiasfactcheck.com/{domain}",
                "summary": f"Source {domain} has documented credibility issues"
            })
        
        # Check content for fact-checkable claims
        if content_summary:
            fact_checks.extend(await self._check_for_existing_fact_checks(content_summary, [url]))
        
        return fact_checks
    
    async def _generate_cross_references(self, content: str, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate cross-references for verification."""
        cross_references = []
        
        # Add general verification resources
        cross_references.extend([
            "https://www.snopes.com",
            "https://www.factcheck.org",
            "https://www.politifact.com"
        ])
        
        # Add topic-specific resources based on content
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["health", "medical", "vaccine", "disease"]):
            cross_references.extend([
                "https://www.who.int",
                "https://www.cdc.gov",
                "https://www.nejm.org"
            ])
        
        if any(word in content_lower for word in ["climate", "environment", "global warming"]):
            cross_references.extend([
                "https://climate.nasa.gov",
                "https://www.ipcc.ch",
                "https://www.nature.com/subjects/climate-change"
            ])
        
        if any(word in content_lower for word in ["election", "voting", "politics"]):
            cross_references.extend([
                "https://www.ballotpedia.org",
                "https://www.fec.gov",
                "https://www.vote.gov"
            ])
        
        return cross_references[:5]  # Limit to 5 references
    
    async def _generate_image_cross_references(self, image_description: str, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate cross-references for image verification."""
        cross_references = [
            "https://www.tineye.com",
            "https://images.google.com",
            "https://www.labnol.org/reverse/"
        ]
        
        # Add image-specific fact-checking resources
        cross_references.extend([
            "https://www.snopes.com/fact-check/",
            "https://www.reuters.com/fact-check/",
            "https://www.ap.org/en-us/topics/politics/elections/how-we-fact-check"
        ])
        
        return cross_references[:5]
    
    async def _generate_url_cross_references(self, url: str, content_analysis: Dict[str, Any]) -> List[str]:
        """Generate cross-references for URL verification."""
        cross_references = []
        
        # Add source evaluation tools
        cross_references.extend([
            "https://mediabiasfactcheck.com",
            "https://www.allsides.com/media-bias",
            "https://adfontesmedia.com/interactive-media-bias-chart/"
        ])
        
        # Add general fact-checking resources
        cross_references.extend([
            "https://www.factcheck.org",
            "https://www.snopes.com",
            "https://www.politifact.com"
        ])
        
        return cross_references[:5]
    
    def _determine_verification_status(
        self, 
        credibility: float, 
        source_assessments: List[Dict], 
        fact_checks: List[Dict]
    ) -> str:
        """Determine overall verification status."""
        if credibility >= 0.8 and not any(fc["verdict"] == "false" for fc in fact_checks):
            return "verified_credible"
        elif credibility <= 0.3 or any(fc["verdict"] == "false" for fc in fact_checks):
            return "verified_problematic"
        elif fact_checks:
            return "partially_verified"
        else:
            return "unverified"
    
    def _determine_image_verification_status(
        self, 
        credibility: float, 
        source_count: int, 
        fact_checks: List[Dict]
    ) -> str:
        """Determine verification status for images."""
        if source_count == 0:
            return "no_sources_found"
        elif credibility >= 0.7 and source_count > 3:
            return "well_sourced"
        elif any(fc["verdict"] == "false" for fc in fact_checks):
            return "debunked"
        elif fact_checks:
            return "fact_checked"
        else:
            return "limited_verification"
    
    def _determine_url_verification_status(
        self, 
        primary_assessment: Dict, 
        fact_checks: List[Dict]
    ) -> str:
        """Determine verification status for URLs."""
        credibility = primary_assessment.get("credibility", 0.5)
        status = primary_assessment.get("status", "unverified")
        
        if status == "verified_credible":
            return "credible_source"
        elif status == "verified_problematic":
            return "problematic_source"
        elif any(fc["verdict"] == "false" for fc in fact_checks):
            return "content_disputed"
        elif fact_checks:
            return "content_fact_checked"
        else:
            return "source_unverified"
    
    def _assess_overall_bias(self, source_assessments: List[Dict]) -> str:
        """Assess overall bias from source assessments."""
        if not source_assessments:
            return "unknown"
        
        biases = [a.get("bias", "unknown") for a in source_assessments]
        bias_counts = {}
        
        for bias in biases:
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
        
        # Return most common bias, or "mixed" if tied
        if bias_counts:
            max_count = max(bias_counts.values())
            most_common = [bias for bias, count in bias_counts.items() if count == max_count]
            
            if len(most_common) == 1:
                return most_common[0]
            else:
                return "mixed"
        
        return "unknown"
    
    def _calculate_verification_confidence(
        self, 
        url_count: int, 
        fact_check_count: int, 
        cross_ref_count: int
    ) -> float:
        """Calculate confidence in verification results."""
        confidence = 0.3  # Base confidence
        
        # More URLs provide more verification opportunities
        confidence += min(0.3, url_count * 0.1)
        
        # Fact-checks increase confidence
        confidence += min(0.3, fact_check_count * 0.15)
        
        # Cross-references provide verification paths
        confidence += min(0.1, cross_ref_count * 0.02)
        
        return min(1.0, confidence)
    
    def _calculate_image_verification_confidence(
        self, 
        reverse_search: Dict, 
        source_assessments: List[Dict], 
        fact_checks: List[Dict]
    ) -> float:
        """Calculate confidence for image verification."""
        confidence = 0.2  # Lower base for images
        
        # Reverse search results increase confidence
        matches = reverse_search.get("matches_found", 0)
        confidence += min(0.4, matches * 0.05)
        
        # Quality of sources matters
        if source_assessments:
            avg_confidence = sum(a.get("confidence", 0.5) for a in source_assessments) / len(source_assessments)
            confidence += avg_confidence * 0.3
        
        # Fact-checks provide strong confidence
        confidence += len(fact_checks) * 0.1
        
        return min(1.0, confidence)
    
    def _calculate_url_verification_confidence(
        self, 
        primary_assessment: Dict, 
        related_assessments: List[Dict], 
        fact_checks: List[Dict]
    ) -> float:
        """Calculate confidence for URL verification."""
        confidence = primary_assessment.get("confidence", 0.5)
        
        # Related sources provide additional confidence
        if related_assessments:
            avg_related_confidence = sum(a.get("confidence", 0.5) for a in related_assessments) / len(related_assessments)
            confidence = (confidence + avg_related_confidence) / 2
        
        # Fact-checks increase confidence
        confidence += len(fact_checks) * 0.1
        
        return min(1.0, confidence)