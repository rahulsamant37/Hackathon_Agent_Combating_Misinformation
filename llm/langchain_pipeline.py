"""
LangChain pipeline system for misinformation analysis.

This module provides reusable LangChain chains for different types of content analysis
including text, image, and URL analysis with structured output parsing.
"""

import json
from typing import Dict, Any, List, Optional, Union
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field

from .llm_router import LLMRouter
from utils.logger import get_logger
from utils.exceptions import ContentProcessingError

logger = get_logger(__name__)


# Pydantic models for structured outputs
class MisinformationIndicator(BaseModel):
    """Model for misinformation indicators."""
    type: str = Field(description="Type of indicator (e.g., emotional_language, logical_fallacy)")
    description: str = Field(description="Description of the indicator")
    severity: float = Field(description="Severity score from 0.0 to 1.0")
    evidence: List[str] = Field(description="Evidence supporting this indicator")
    explanation: str = Field(description="Detailed explanation of why this is problematic")


class SourceInfo(BaseModel):
    """Model for source credibility information."""
    domain: Optional[str] = Field(description="Source domain name")
    credibility_score: float = Field(description="Credibility score from 0.0 to 1.0")
    bias_rating: Optional[str] = Field(description="Bias rating (left, center, right)")
    reputation: str = Field(description="Overall reputation assessment")
    verification_status: str = Field(description="Verification status of the source")


class TextAnalysisResult(BaseModel):
    """Model for text analysis results."""
    risk_score: float = Field(description="Overall risk score from 0.0 to 1.0")
    risk_level: str = Field(description="Risk level: Low, Medium, or High")
    confidence: float = Field(description="Confidence in the analysis from 0.0 to 1.0")
    indicators: List[MisinformationIndicator] = Field(description="List of misinformation indicators found")
    summary: str = Field(description="Summary of the analysis")
    recommendations: List[str] = Field(description="Recommendations for the user")


class ImageAnalysisResult(BaseModel):
    """Model for image analysis results."""
    risk_score: float = Field(description="Overall risk score from 0.0 to 1.0")
    risk_level: str = Field(description="Risk level: Low, Medium, or High")
    confidence: float = Field(description="Confidence in the analysis from 0.0 to 1.0")
    manipulation_detected: bool = Field(description="Whether manipulation was detected")
    manipulation_types: List[str] = Field(description="Types of manipulation detected")
    technical_indicators: List[str] = Field(description="Technical indicators of manipulation")
    summary: str = Field(description="Summary of the analysis")
    recommendations: List[str] = Field(description="Recommendations for the user")


class URLAnalysisResult(BaseModel):
    """Model for URL analysis results."""
    risk_score: float = Field(description="Overall risk score from 0.0 to 1.0")
    risk_level: str = Field(description="Risk level: Low, Medium, or High")
    confidence: float = Field(description="Confidence in the analysis from 0.0 to 1.0")
    source_info: SourceInfo = Field(description="Information about the source")
    content_indicators: List[MisinformationIndicator] = Field(description="Content-based indicators")
    summary: str = Field(description="Summary of the analysis")
    recommendations: List[str] = Field(description="Recommendations for the user")


class LangChainPipeline:
    """
    LangChain pipeline system for misinformation analysis.
    
    Provides reusable chains for analyzing different types of content
    with structured output parsing and error handling.
    """
    
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        self.chains = {}
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize all analysis chains."""
        logger.info("Initializing LangChain analysis pipelines")
        
        # Initialize parsers
        self.text_parser = PydanticOutputParser(pydantic_object=TextAnalysisResult)
        self.image_parser = PydanticOutputParser(pydantic_object=ImageAnalysisResult)
        self.url_parser = PydanticOutputParser(pydantic_object=URLAnalysisResult)
        
        # Create chains
        self.chains["text_analysis"] = self._create_text_analysis_chain()
        self.chains["image_analysis"] = self._create_image_analysis_chain()
        self.chains["url_analysis"] = self._create_url_analysis_chain()
        
        logger.info("LangChain pipelines initialized successfully")
    
    def _create_text_analysis_chain(self) -> ChatPromptTemplate:
        """Create text analysis chain with prompt template."""
        system_template = """You are an expert misinformation analyst. Your task is to analyze text content for potential misinformation indicators.

Consider the following factors in your analysis:
1. Emotional language and manipulation tactics
2. Logical fallacies and reasoning errors
3. Factual claims that can be verified
4. Source credibility indicators
5. Bias and propaganda techniques
6. Sensationalism and clickbait elements

Provide a thorough analysis with specific evidence and explanations.

{format_instructions}"""

        human_template = """Please analyze the following text for misinformation indicators:

Text to analyze:
{text_content}

Additional context (if any):
{context}

Provide a detailed analysis following the specified format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_image_analysis_chain(self) -> ChatPromptTemplate:
        """Create image analysis chain for manipulation detection."""
        system_template = """You are an expert in digital image forensics and manipulation detection. Your task is to analyze images for signs of manipulation or misinformation.

Consider the following factors:
1. Technical inconsistencies (lighting, shadows, compression artifacts)
2. Visual anomalies and impossible elements
3. Metadata inconsistencies
4. Common manipulation techniques (deepfakes, photoshop, etc.)
5. Context and plausibility of the image content
6. Reverse image search results (if provided)

Provide specific technical indicators and explanations for your findings.

{format_instructions}"""

        human_template = """Please analyze the following image for manipulation or misinformation indicators:

Image description: {image_description}
Image metadata: {image_metadata}
Reverse image search results: {reverse_search_results}
Additional context: {context}

Provide a detailed technical analysis following the specified format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_url_analysis_chain(self) -> ChatPromptTemplate:
        """Create URL analysis chain for source credibility assessment."""
        system_template = """You are an expert in source credibility assessment and media literacy. Your task is to analyze URLs and their content for reliability and potential misinformation.

Consider the following factors:
1. Domain reputation and history
2. Website design and professionalism
3. About page and transparency
4. Editorial standards and fact-checking
5. Funding sources and potential conflicts of interest
6. Content quality and accuracy
7. Bias indicators and political leanings

Provide a comprehensive assessment of source credibility and content reliability.

{format_instructions}"""

        human_template = """Please analyze the following URL and its content for credibility and misinformation risk:

URL: {url}
Domain information: {domain_info}
Content summary: {content_summary}
Author information: {author_info}
Publication date: {publication_date}
Additional context: {context}

Provide a detailed credibility assessment following the specified format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def analyze_text(
        self,
        text_content: str,
        context: Optional[str] = None,
        provider: Optional[str] = None
    ) -> TextAnalysisResult:
        """
        Analyze text content for misinformation indicators.
        
        Args:
            text_content: The text to analyze
            context: Additional context about the text
            provider: Specific LLM provider to use
            
        Returns:
            TextAnalysisResult: Structured analysis results
        """
        try:
            logger.info("Starting text analysis")
            
            # Prepare the prompt
            chain = self.chains["text_analysis"]
            messages = chain.format_messages(
                text_content=text_content,
                context=context or "No additional context provided",
                format_instructions=self.text_parser.get_format_instructions()
            )
            
            # Generate response
            response = await self.llm_router.generate_structured_response(
                messages=messages,
                response_schema=TextAnalysisResult.schema(),
                provider=provider
            )
            
            # Parse the response
            try:
                if response.metadata.get("schema_valid", False):
                    parsed_result = TextAnalysisResult(**response.metadata["parsed_content"])
                else:
                    # Try to parse the raw content
                    parsed_result = self.text_parser.parse(response.content)
                
                logger.info(f"Text analysis completed with risk level: {parsed_result.risk_level}")
                return parsed_result
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured response, using fallback: {str(parse_error)}")
                # Create a fallback response
                return self._create_fallback_text_result(response.content, text_content)
                
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise ContentProcessingError(f"Text analysis failed: {str(e)}")
    
    async def analyze_image(
        self,
        image_description: str,
        image_metadata: Optional[Dict[str, Any]] = None,
        reverse_search_results: Optional[str] = None,
        context: Optional[str] = None,
        provider: Optional[str] = None
    ) -> ImageAnalysisResult:
        """
        Analyze image for manipulation indicators.
        
        Args:
            image_description: Description of the image content
            image_metadata: Image metadata information
            reverse_search_results: Results from reverse image search
            context: Additional context about the image
            provider: Specific LLM provider to use
            
        Returns:
            ImageAnalysisResult: Structured analysis results
        """
        try:
            logger.info("Starting image analysis")
            
            # Prepare the prompt
            chain = self.chains["image_analysis"]
            messages = chain.format_messages(
                image_description=image_description,
                image_metadata=json.dumps(image_metadata or {}, indent=2),
                reverse_search_results=reverse_search_results or "No reverse search results available",
                context=context or "No additional context provided",
                format_instructions=self.image_parser.get_format_instructions()
            )
            
            # Generate response
            response = await self.llm_router.generate_structured_response(
                messages=messages,
                response_schema=ImageAnalysisResult.schema(),
                provider=provider
            )
            
            # Parse the response
            try:
                if response.metadata.get("schema_valid", False):
                    parsed_result = ImageAnalysisResult(**response.metadata["parsed_content"])
                else:
                    parsed_result = self.image_parser.parse(response.content)
                
                logger.info(f"Image analysis completed with risk level: {parsed_result.risk_level}")
                return parsed_result
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured response, using fallback: {str(parse_error)}")
                return self._create_fallback_image_result(response.content, image_description)
                
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise ContentProcessingError(f"Image analysis failed: {str(e)}")
    
    async def analyze_url(
        self,
        url: str,
        domain_info: Optional[Dict[str, Any]] = None,
        content_summary: Optional[str] = None,
        author_info: Optional[str] = None,
        publication_date: Optional[str] = None,
        context: Optional[str] = None,
        provider: Optional[str] = None
    ) -> URLAnalysisResult:
        """
        Analyze URL and content for credibility and misinformation risk.
        
        Args:
            url: The URL to analyze
            domain_info: Information about the domain
            content_summary: Summary of the content
            author_info: Information about the author
            publication_date: Publication date of the content
            context: Additional context
            provider: Specific LLM provider to use
            
        Returns:
            URLAnalysisResult: Structured analysis results
        """
        try:
            logger.info(f"Starting URL analysis for: {url}")
            
            # Prepare the prompt
            chain = self.chains["url_analysis"]
            messages = chain.format_messages(
                url=url,
                domain_info=json.dumps(domain_info or {}, indent=2),
                content_summary=content_summary or "No content summary available",
                author_info=author_info or "No author information available",
                publication_date=publication_date or "No publication date available",
                context=context or "No additional context provided",
                format_instructions=self.url_parser.get_format_instructions()
            )
            
            # Generate response
            response = await self.llm_router.generate_structured_response(
                messages=messages,
                response_schema=URLAnalysisResult.schema(),
                provider=provider
            )
            
            # Parse the response
            try:
                if response.metadata.get("schema_valid", False):
                    parsed_result = URLAnalysisResult(**response.metadata["parsed_content"])
                else:
                    parsed_result = self.url_parser.parse(response.content)
                
                logger.info(f"URL analysis completed with risk level: {parsed_result.risk_level}")
                return parsed_result
                
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured response, using fallback: {str(parse_error)}")
                return self._create_fallback_url_result(response.content, url)
                
        except Exception as e:
            logger.error(f"URL analysis failed: {str(e)}")
            raise ContentProcessingError(f"URL analysis failed: {str(e)}")
    
    def _create_fallback_text_result(self, raw_response: str, text_content: str) -> TextAnalysisResult:
        """Create a fallback text analysis result when parsing fails."""
        return TextAnalysisResult(
            risk_score=0.5,
            risk_level="Medium",
            confidence=0.3,
            indicators=[
                MisinformationIndicator(
                    type="parsing_error",
                    description="Analysis parsing failed",
                    severity=0.5,
                    evidence=["Unable to parse structured response"],
                    explanation="The analysis could not be properly parsed. Manual review recommended."
                )
            ],
            summary=f"Analysis completed but parsing failed. Raw response: {raw_response[:200]}...",
            recommendations=["Manual review recommended due to parsing error"]
        )
    
    def _create_fallback_image_result(self, raw_response: str, image_description: str) -> ImageAnalysisResult:
        """Create a fallback image analysis result when parsing fails."""
        return ImageAnalysisResult(
            risk_score=0.5,
            risk_level="Medium",
            confidence=0.3,
            manipulation_detected=False,
            manipulation_types=[],
            technical_indicators=["Parsing error occurred"],
            summary=f"Analysis completed but parsing failed. Raw response: {raw_response[:200]}...",
            recommendations=["Manual review recommended due to parsing error"]
        )
    
    def _create_fallback_url_result(self, raw_response: str, url: str) -> URLAnalysisResult:
        """Create a fallback URL analysis result when parsing fails."""
        return URLAnalysisResult(
            risk_score=0.5,
            risk_level="Medium",
            confidence=0.3,
            source_info=SourceInfo(
                domain=url,
                credibility_score=0.5,
                bias_rating="unknown",
                reputation="Unable to assess",
                verification_status="Error in analysis"
            ),
            content_indicators=[
                MisinformationIndicator(
                    type="parsing_error",
                    description="Analysis parsing failed",
                    severity=0.5,
                    evidence=["Unable to parse structured response"],
                    explanation="The analysis could not be properly parsed. Manual review recommended."
                )
            ],
            summary=f"Analysis completed but parsing failed. Raw response: {raw_response[:200]}...",
            recommendations=["Manual review recommended due to parsing error"]
        )