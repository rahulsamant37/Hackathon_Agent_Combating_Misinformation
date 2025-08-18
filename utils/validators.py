"""
Input validation utilities for the misinformation detection tool.

This module provides comprehensive validation for text content, URLs,
and content type detection with security and format checks.
"""

import re
import urllib.parse
import validators as external_validators
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse, urljoin
import requests
from datetime import datetime
import html
import unicodedata
from io import BytesIO

from config.settings import get_config
from utils.logger import get_logger
from utils.exceptions import ValidationError, ExternalServiceError

logger = get_logger(__name__)


class TextValidator:
    """Validates text content for format, length, and security."""
    
    # Suspicious patterns that might indicate malicious content
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript
        r'javascript:',  # JavaScript URLs
        r'vbscript:',  # VBScript URLs
        r'data:text/html',  # Data URLs with HTML
        r'<iframe[^>]*>',  # Iframes
        r'<object[^>]*>',  # Objects
        r'<embed[^>]*>',  # Embeds
        r'<form[^>]*>',  # Forms
        r'<input[^>]*>',  # Input fields
    ]
    
    # Common encoding attacks
    ENCODING_ATTACKS = [
        r'%3Cscript%3E',  # URL encoded script
        r'&lt;script&gt;',  # HTML encoded script
        r'\\u003cscript\\u003e',  # Unicode encoded script
        r'&#x3C;script&#x3E;',  # Hex encoded script
    ]
    
    def __init__(self):
        """Initialize the text validator."""
        self.config = get_config()
        self.max_length = self.config.analysis.text.max_length
    
    def validate_length(self, text: str, max_length: Optional[int] = None) -> None:
        """Validate text length against limits.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length (uses config default if None)
            
        Raises:
            ValidationError: If text length is invalid
        """
        if not isinstance(text, str):
            raise ValidationError(
                "Text must be a string",
                field="text_type",
                value=type(text).__name__
            )
        
        max_len = max_length or self.max_length
        
        if len(text) == 0:
            raise ValidationError(
                "Text cannot be empty",
                field="text_length",
                value=0
            )
        
        if len(text) > max_len:
            raise ValidationError(
                f"Text length {len(text)} exceeds maximum allowed length {max_len}",
                field="text_length",
                value=len(text)
            )
    
    def validate_encoding(self, text: str) -> str:
        """Validate and normalize text encoding.
        
        Args:
            text: Text to validate
            
        Returns:
            Normalized text
            
        Raises:
            ValidationError: If text contains invalid characters
        """
        try:
            # Normalize Unicode characters
            normalized = unicodedata.normalize('NFKC', text)
            
            # Check for null bytes and other control characters
            if '\x00' in normalized:
                raise ValidationError(
                    "Text contains null bytes",
                    field="text_encoding"
                )
            
            # Remove or replace problematic control characters
            # Keep only printable characters, whitespace, and common punctuation
            cleaned = ''.join(
                char for char in normalized 
                if unicodedata.category(char)[0] not in ['C'] or char in ['\n', '\r', '\t']
            )
            
            return cleaned
            
        except Exception as e:
            raise ValidationError(
                f"Text encoding validation failed: {str(e)}",
                field="text_encoding"
            )
    
    def check_suspicious_content(self, text: str) -> List[Dict[str, str]]:
        """Check for suspicious patterns in text content.
        
        Args:
            text: Text to check
            
        Returns:
            List of detected suspicious patterns
        """
        suspicious_findings = []
        
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                suspicious_findings.append({
                    'type': 'suspicious_pattern',
                    'pattern': pattern,
                    'match': match.group()[:100],  # Truncate long matches
                    'position': match.start()
                })
        
        # Check for encoding attacks
        for pattern in self.ENCODING_ATTACKS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                suspicious_findings.append({
                    'type': 'encoding_attack',
                    'pattern': pattern,
                    'match': match.group(),
                    'position': match.start()
                })
        
        return suspicious_findings
    
    def sanitize_text(self, text: str, strict: bool = False) -> str:
        """Sanitize text content for safe processing.
        
        Args:
            text: Text to sanitize
            strict: Whether to apply strict sanitization
            
        Returns:
            Sanitized text
        """
        # First normalize encoding
        sanitized = self.validate_encoding(text)
        
        if strict:
            # HTML escape all content
            sanitized = html.escape(sanitized)
            
            # Remove potentially dangerous patterns
            for pattern in self.SUSPICIOUS_PATTERNS:
                sanitized = re.sub(pattern, '[REMOVED]', sanitized, flags=re.IGNORECASE | re.DOTALL)
        else:
            # Basic sanitization - remove null bytes and control characters
            sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text metadata
        """
        lines = text.split('\n')
        words = text.split()
        
        # Count different character types
        char_counts = {
            'letters': sum(1 for c in text if c.isalpha()),
            'digits': sum(1 for c in text if c.isdigit()),
            'spaces': sum(1 for c in text if c.isspace()),
            'punctuation': sum(1 for c in text if not c.isalnum() and not c.isspace()),
        }
        
        # Detect language hints (basic)
        language_hints = []
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            language_hints.append('russian')
        if re.search(r'[α-ωάέήίόύώ]', text, re.IGNORECASE):
            language_hints.append('greek')
        if re.search(r'[äöüß]', text, re.IGNORECASE):
            language_hints.append('german')
        if re.search(r'[àâäéèêëïîôöùûüÿç]', text, re.IGNORECASE):
            language_hints.append('french')
        if re.search(r'[áéíñóúü]', text, re.IGNORECASE):
            language_hints.append('spanish')
        
        # Check for URLs and email addresses
        url_count = len(re.findall(r'https?://[^\s]+', text))
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        return {
            'length': len(text),
            'lines': len(lines),
            'words': len(words),
            'characters': char_counts,
            'language_hints': language_hints,
            'url_count': url_count,
            'email_count': email_count,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / max(text.count('.') + text.count('!') + text.count('?'), 1)
        }
    
    def validate_text(self, text: str, strict: bool = False) -> Dict[str, Any]:
        """Comprehensive text validation.
        
        Args:
            text: Text to validate
            strict: Whether to apply strict validation
            
        Returns:
            Validation results with metadata
            
        Raises:
            ValidationError: If validation fails
        """
        # Basic validations
        self.validate_length(text)
        normalized_text = self.validate_encoding(text)
        
        # Security checks
        suspicious_content = self.check_suspicious_content(normalized_text)
        
        if strict and suspicious_content:
            raise ValidationError(
                f"Text contains {len(suspicious_content)} suspicious patterns",
                field="suspicious_content",
                details={'patterns': suspicious_content}
            )
        
        # Extract metadata
        metadata = self.extract_metadata(normalized_text)
        
        # Sanitize if needed
        sanitized_text = self.sanitize_text(normalized_text, strict)
        
        return {
            'valid': True,
            'original_text': text,
            'normalized_text': normalized_text,
            'sanitized_text': sanitized_text,
            'metadata': metadata,
            'suspicious_content': suspicious_content,
            'validation_time': datetime.now()
        }


class URLValidator:
    """Validates and analyzes URLs for security and accessibility."""
    
    # Dangerous URL schemes
    DANGEROUS_SCHEMES = [
        'javascript', 'vbscript', 'data', 'file', 'ftp'
    ]
    
    # Suspicious URL patterns
    SUSPICIOUS_PATTERNS = [
        r'bit\.ly',  # URL shorteners (can hide destination)
        r'tinyurl\.com',
        r't\.co',
        r'goo\.gl',
        r'ow\.ly',
        r'short\.link',
        r'tinycc\.com',
        r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
        r'localhost',
        r'127\.0\.0\.1',
        r'0\.0\.0\.0',
        r'10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # Private IP ranges
        r'192\.168\.[0-9]{1,3}\.[0-9]{1,3}',
        r'172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.[0-9]{1,3}',
    ]
    
    def __init__(self):
        """Initialize the URL validator."""
        self.config = get_config()
        self.timeout = self.config.analysis.url.timeout
        self.max_redirects = self.config.analysis.url.max_redirects
        self.user_agent = self.config.analysis.url.user_agent
    
    def validate_url_format(self, url: str) -> str:
        """Validate URL format and structure.
        
        Args:
            url: URL to validate
            
        Returns:
            Normalized URL
            
        Raises:
            ValidationError: If URL format is invalid
        """
        if not isinstance(url, str):
            raise ValidationError(
                "URL must be a string",
                field="url_type",
                value=type(url).__name__
            )
        
        if not url.strip():
            raise ValidationError(
                "URL cannot be empty",
                field="url",
                value=url
            )
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Validate using external validator
        if not external_validators.url(url):
            raise ValidationError(
                f"Invalid URL format: {url}",
                field="url_format",
                value=url
            )
        
        return url
    
    def check_url_security(self, url: str) -> List[Dict[str, str]]:
        """Check URL for security issues.
        
        Args:
            url: URL to check
            
        Returns:
            List of security issues found
        """
        security_issues = []
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme.lower() in self.DANGEROUS_SCHEMES:
            security_issues.append({
                'type': 'dangerous_scheme',
                'description': f"Dangerous URL scheme: {parsed.scheme}",
                'severity': 'high'
            })
        
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                security_issues.append({
                    'type': 'suspicious_pattern',
                    'description': f"Suspicious URL pattern detected: {pattern}",
                    'severity': 'medium'
                })
        
        # Check for non-standard ports
        if parsed.port and parsed.port not in [80, 443, 8080, 8443]:
            security_issues.append({
                'type': 'non_standard_port',
                'description': f"Non-standard port: {parsed.port}",
                'severity': 'low'
            })
        
        # Check for very long URLs (potential attack)
        if len(url) > 2048:
            security_issues.append({
                'type': 'long_url',
                'description': f"Unusually long URL: {len(url)} characters",
                'severity': 'medium'
            })
        
        return security_issues
    
    def fetch_url_metadata(self, url: str) -> Dict[str, Any]:
        """Fetch metadata from URL without downloading full content.
        
        Args:
            url: URL to fetch metadata from
            
        Returns:
            Dictionary with URL metadata
            
        Raises:
            ExternalServiceError: If URL cannot be accessed
        """
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # Make HEAD request first to get headers
            response = requests.head(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=True
            )
            
            metadata = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': response.headers.get('content-length'),
                'server': response.headers.get('server', ''),
                'last_modified': response.headers.get('last-modified'),
                'final_url': response.url,
                'redirect_count': len(response.history),
                'ssl_verified': response.url.startswith('https://'),
            }
            
            # If it's HTML, try to get more metadata with a GET request
            if 'text/html' in metadata['content_type']:
                try:
                    # Limit content size
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=True,
                        verify=True,
                        stream=True
                    )
                    
                    # Read only first 8KB for title extraction
                    content = b''
                    for chunk in response.iter_content(chunk_size=1024):
                        content += chunk
                        if len(content) > 8192:  # 8KB limit
                            break
                    
                    # Try to extract title
                    content_str = content.decode('utf-8', errors='ignore')
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', content_str, re.IGNORECASE | re.DOTALL)
                    if title_match:
                        metadata['title'] = title_match.group(1).strip()[:200]  # Limit title length
                    
                except Exception as e:
                    logger.warning(f"Could not fetch HTML metadata for {url}: {str(e)}")
            
            return metadata
            
        except requests.exceptions.Timeout:
            raise ExternalServiceError(
                f"URL request timed out after {self.timeout} seconds",
                service_name="url_fetch",
                status_code=408
            )
        except requests.exceptions.ConnectionError:
            raise ExternalServiceError(
                f"Could not connect to URL: {url}",
                service_name="url_fetch"
            )
        except requests.exceptions.SSLError:
            raise ExternalServiceError(
                f"SSL certificate verification failed for: {url}",
                service_name="url_fetch"
            )
        except Exception as e:
            raise ExternalServiceError(
                f"Error fetching URL metadata: {str(e)}",
                service_name="url_fetch"
            )
    
    def validate_url(self, url: str, fetch_metadata: bool = True) -> Dict[str, Any]:
        """Comprehensive URL validation.
        
        Args:
            url: URL to validate
            fetch_metadata: Whether to fetch metadata from the URL
            
        Returns:
            Validation results with metadata
            
        Raises:
            ValidationError: If validation fails
        """
        # Format validation
        normalized_url = self.validate_url_format(url)
        parsed = urlparse(normalized_url)
        
        # Security checks
        security_issues = self.check_url_security(normalized_url)
        
        result = {
            'valid': True,
            'original_url': url,
            'normalized_url': normalized_url,
            'parsed_url': {
                'scheme': parsed.scheme,
                'netloc': parsed.netloc,
                'path': parsed.path,
                'params': parsed.params,
                'query': parsed.query,
                'fragment': parsed.fragment
            },
            'security_issues': security_issues,
            'validation_time': datetime.now()
        }
        
        # Fetch metadata if requested
        if fetch_metadata:
            try:
                metadata = self.fetch_url_metadata(normalized_url)
                result['metadata'] = metadata
            except ExternalServiceError as e:
                result['metadata_error'] = str(e)
                logger.warning(f"Could not fetch metadata for {normalized_url}: {str(e)}")
        
        return result


class ContentTypeDetector:
    """Detects and classifies content types from various inputs."""
    
    def __init__(self):
        """Initialize the content type detector."""
        self.text_validator = TextValidator()
        self.url_validator = URLValidator()
    
    def detect_content_type(self, content: str) -> Dict[str, Any]:
        """Detect the type of content from input string.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with content type and confidence
        """
        content = content.strip()
        
        if not content:
            return {
                'type': 'empty',
                'confidence': 1.0,
                'details': 'Content is empty'
            }
        
        # Check if it's a URL
        url_patterns = [
            r'^https?://',
            r'^www\.',
            r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}',
        ]
        
        for pattern in url_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return {
                    'type': 'url',
                    'confidence': 0.9,
                    'details': f'Matches URL pattern: {pattern}'
                }
        
        # Check if it looks like HTML
        html_indicators = [
            r'<html[^>]*>',
            r'<head[^>]*>',
            r'<body[^>]*>',
            r'<div[^>]*>',
            r'<p[^>]*>',
            r'<script[^>]*>',
        ]
        
        html_matches = sum(1 for pattern in html_indicators if re.search(pattern, content, re.IGNORECASE))
        if html_matches >= 2:
            return {
                'type': 'html',
                'confidence': min(0.9, 0.3 + html_matches * 0.15),
                'details': f'Contains {html_matches} HTML indicators'
            }
        
        # Check if it's structured data (JSON, XML, etc.)
        if content.strip().startswith(('{', '[')):
            try:
                import json
                json.loads(content)
                return {
                    'type': 'json',
                    'confidence': 0.95,
                    'details': 'Valid JSON format'
                }
            except:
                pass
        
        if content.strip().startswith('<?xml') or content.strip().startswith('<'):
            return {
                'type': 'xml',
                'confidence': 0.8,
                'details': 'XML-like structure detected'
            }
        
        # Default to plain text
        return {
            'type': 'text',
            'confidence': 0.7,
            'details': 'Appears to be plain text content'
        }
    
    def validate_content(
        self, 
        content: str, 
        expected_type: Optional[str] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Validate content based on detected or expected type.
        
        Args:
            content: Content to validate
            expected_type: Expected content type (auto-detect if None)
            strict: Whether to apply strict validation
            
        Returns:
            Comprehensive validation results
        """
        # Detect content type if not specified
        if expected_type is None:
            type_detection = self.detect_content_type(content)
            content_type = type_detection['type']
        else:
            content_type = expected_type
            type_detection = {'type': content_type, 'confidence': 1.0, 'details': 'User specified'}
        
        result = {
            'content_type': content_type,
            'type_detection': type_detection,
            'validation_time': datetime.now()
        }
        
        # Validate based on content type
        try:
            if content_type == 'url':
                validation_result = self.url_validator.validate_url(content, fetch_metadata=True)
                result.update(validation_result)
            
            elif content_type in ['text', 'html', 'json', 'xml']:
                validation_result = self.text_validator.validate_text(content, strict=strict)
                result.update(validation_result)
            
            else:
                # Generic text validation for unknown types
                validation_result = self.text_validator.validate_text(content, strict=False)
                result.update(validation_result)
            
            result['validation_success'] = True
            
        except (ValidationError, ExternalServiceError) as e:
            result['validation_success'] = False
            result['validation_error'] = str(e)
            result['error_details'] = e.to_dict() if hasattr(e, 'to_dict') else {}
        
        return result


# Convenience functions for common validation tasks

def validate_text_input(
    text: str, 
    max_length: Optional[int] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """Validate text input with comprehensive checks.
    
    Args:
        text: Text to validate
        max_length: Maximum allowed length
        strict: Whether to apply strict validation
        
    Returns:
        Validation results
    """
    validator = TextValidator()
    return validator.validate_text(text, strict=strict)


def validate_url_input(url: str, fetch_metadata: bool = True) -> Dict[str, Any]:
    """Validate URL input with security checks.
    
    Args:
        url: URL to validate
        fetch_metadata: Whether to fetch metadata
        
    Returns:
        Validation results
    """
    validator = URLValidator()
    return validator.validate_url(url, fetch_metadata=fetch_metadata)


def detect_and_validate_content(
    content: str,
    expected_type: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """Detect content type and validate accordingly.
    
    Args:
        content: Content to validate
        expected_type: Expected content type
        strict: Whether to apply strict validation
        
    Returns:
        Validation results with type detection
    """
    detector = ContentTypeDetector()
    return detector.validate_content(content, expected_type, strict)


def validate_image_format(filename: str, file_content: bytes) -> Dict[str, Any]:
    """Validate image format and basic properties.
    
    Args:
        filename: Original filename
        file_content: Image file content as bytes
        
    Returns:
        Validation results with image metadata
        
    Raises:
        ValidationError: If image validation fails
    """
    from utils.file_utils import validate_uploaded_file
    from io import BytesIO
    
    # Use file utils to validate the image
    return validate_uploaded_file(
        file_content=BytesIO(file_content),
        filename=filename,
        allowed_categories=['image']
    )


def sanitize_url(url: str) -> str:
    """Sanitize URL for safe processing.
    
    Args:
        url: URL to sanitize
        
    Returns:
        Sanitized URL
        
    Raises:
        ValidationError: If URL cannot be sanitized safely
    """
    validator = URLValidator()
    
    # Basic format validation and normalization
    normalized_url = validator.validate_url_format(url)
    
    # Check for security issues
    security_issues = validator.check_url_security(normalized_url)
    
    # Block URLs with high-severity security issues
    high_severity_issues = [issue for issue in security_issues if issue.get('severity') == 'high']
    if high_severity_issues:
        raise ValidationError(
            f"URL contains high-severity security issues: {[issue['description'] for issue in high_severity_issues]}",
            field="url_security",
            value=url
        )
    
    return normalized_url


def validate_content_length(content: str, content_type: str) -> None:
    """Validate content length based on type.
    
    Args:
        content: Content to validate
        content_type: Type of content (text, url, etc.)
        
    Raises:
        ValidationError: If content length is invalid
    """
    config = get_config()
    
    if content_type == 'text':
        max_length = config.analysis.text.max_length
        if len(content) > max_length:
            raise ValidationError(
                f"Text content exceeds maximum length of {max_length} characters",
                field="content_length",
                value=len(content)
            )
    elif content_type == 'url':
        # URLs should be reasonable length
        if len(content) > 2048:
            raise ValidationError(
                f"URL exceeds maximum length of 2048 characters",
                field="url_length", 
                value=len(content)
            )
    
    if len(content.strip()) == 0:
        raise ValidationError(
            "Content cannot be empty",
            field="content_empty",
            value=content
        )


def extract_content_features(content: str, content_type: str) -> Dict[str, Any]:
    """Extract features from content for analysis.
    
    Args:
        content: Content to analyze
        content_type: Type of content
        
    Returns:
        Dictionary with extracted features
    """
    features = {
        'content_type': content_type,
        'length': len(content),
        'word_count': len(content.split()) if content_type == 'text' else 0,
        'extraction_time': datetime.now()
    }
    
    if content_type == 'text':
        validator = TextValidator()
        metadata = validator.extract_metadata(content)
        features.update(metadata)
        
        # Add additional text features
        features.update({
            'has_urls': metadata['url_count'] > 0,
            'has_emails': metadata['email_count'] > 0,
            'avg_word_length': metadata['avg_word_length'],
            'sentence_count': max(content.count('.') + content.count('!') + content.count('?'), 1),
            'exclamation_count': content.count('!'),
            'question_count': content.count('?'),
            'uppercase_ratio': sum(1 for c in content if c.isupper()) / len(content) if content else 0,
            'digit_ratio': sum(1 for c in content if c.isdigit()) / len(content) if content else 0,
        })
    
    elif content_type == 'url':
        from urllib.parse import urlparse
        parsed = urlparse(content)
        features.update({
            'domain': parsed.netloc,
            'path_segments': len([p for p in parsed.path.split('/') if p]),
            'has_query': bool(parsed.query),
            'has_fragment': bool(parsed.fragment),
            'is_https': parsed.scheme == 'https',
            'subdomain_count': len(parsed.netloc.split('.')) - 2 if parsed.netloc else 0,
        })
    
    return features


def comprehensive_input_validation(
    content: str,
    content_type: Optional[str] = None,
    strict: bool = False,
    extract_features: bool = True
) -> Dict[str, Any]:
    """Perform comprehensive validation on input content.
    
    This is the main validation function that combines all validation capabilities.
    
    Args:
        content: Content to validate
        content_type: Expected content type (auto-detect if None)
        strict: Whether to apply strict validation
        extract_features: Whether to extract content features
        
    Returns:
        Comprehensive validation results
        
    Raises:
        ValidationError: If validation fails
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Content type detection
        detector = ContentTypeDetector()
        if content_type is None:
            type_detection = detector.detect_content_type(content)
            detected_type = type_detection['type']
        else:
            detected_type = content_type
            type_detection = {'type': content_type, 'confidence': 1.0, 'details': 'User specified'}
        
        # Step 2: Basic content validation
        validate_content_length(content, detected_type)
        
        # Step 3: Type-specific validation
        validation_result = detector.validate_content(content, detected_type, strict)
        
        # Step 4: Feature extraction
        features = {}
        if extract_features:
            features = extract_content_features(content, detected_type)
        
        # Step 5: Compile comprehensive results
        result = {
            'validation_success': True,
            'content_type': detected_type,
            'type_detection': type_detection,
            'validation_time': datetime.now() - start_time,
            'strict_mode': strict,
            'features': features,
        }
        
        # Merge type-specific validation results
        result.update(validation_result)
        
        return result
        
    except (ValidationError, ExternalServiceError) as e:
        # Re-raise validation errors with additional context
        raise ValidationError(
            f"Input validation failed: {str(e)}",
            field="comprehensive_validation",
            details={
                'original_error': str(e),
                'content_type': content_type,
                'strict_mode': strict,
                'validation_time': datetime.now() - start_time
            }
        )
    except Exception as e:
        # Wrap unexpected errors
        raise ValidationError(
            f"Unexpected error during validation: {str(e)}",
            field="validation_system",
            details={
                'error_type': type(e).__name__,
                'validation_time': datetime.now() - start_time
            }
        )