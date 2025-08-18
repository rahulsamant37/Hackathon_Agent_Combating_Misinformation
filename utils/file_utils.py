"""
File handling utilities for the misinformation detection tool.

This module provides secure file upload validation, storage, and cleanup
mechanisms for handling user-submitted content.
"""

import os
import hashlib
import mimetypes
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from datetime import datetime, timedelta
import magic
from PIL import Image
import uuid

from config.settings import get_config
from utils.logger import get_logger
from utils.exceptions import (
    FileProcessingError, 
    ValidationError,
    configuration_error
)

logger = get_logger(__name__)


class FileValidator:
    """Validates uploaded files for security and format compliance."""
    
    # Allowed MIME types for different content categories
    ALLOWED_MIME_TYPES = {
        'image': [
            'image/jpeg',
            'image/jpg', 
            'image/png',
            'image/gif',
            'image/webp',
            'image/bmp',
            'image/tiff'
        ],
        'text': [
            'text/plain',
            'text/html',
            'text/markdown',
            'text/csv',
            'application/json',
            'application/xml'
        ],
        'document': [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/rtf'
        ]
    }
    
    # File extensions mapping
    ALLOWED_EXTENSIONS = {
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'],
        'text': ['.txt', '.html', '.htm', '.md', '.csv', '.json', '.xml'],
        'document': ['.pdf', '.doc', '.docx', '.rtf']
    }
    
    # Dangerous file signatures to block
    DANGEROUS_SIGNATURES = [
        b'\x4d\x5a',  # PE executable
        b'\x7f\x45\x4c\x46',  # ELF executable
        b'\xca\xfe\xba\xbe',  # Mach-O executable
        b'\x50\x4b\x03\x04',  # ZIP (could contain executables)
    ]
    
    def __init__(self):
        """Initialize the file validator."""
        self.config = get_config()
        self.max_file_size = self.config.analysis.image.max_size
        self.allowed_formats = self.config.analysis.image.supported_formats
    
    def validate_file_size(self, file_size: int, max_size: Optional[int] = None) -> None:
        """Validate file size against limits.
        
        Args:
            file_size: Size of the file in bytes
            max_size: Maximum allowed size (uses config default if None)
            
        Raises:
            ValidationError: If file size exceeds limits
        """
        max_allowed = max_size or self.max_file_size
        
        if file_size > max_allowed:
            raise ValidationError(
                f"File size {file_size} bytes exceeds maximum allowed size {max_allowed} bytes",
                field="file_size",
                value=file_size
            )
        
        if file_size <= 0:
            raise ValidationError(
                "File size must be greater than 0 bytes",
                field="file_size", 
                value=file_size
            )
    
    def validate_file_extension(self, filename: str, allowed_categories: List[str] = None) -> str:
        """Validate file extension against allowed types.
        
        Args:
            filename: Name of the file
            allowed_categories: List of allowed categories (image, text, document)
            
        Returns:
            The detected file category
            
        Raises:
            ValidationError: If file extension is not allowed
        """
        if not filename:
            raise ValidationError("Filename cannot be empty", field="filename")
        
        file_ext = Path(filename).suffix.lower()
        
        if not file_ext:
            raise ValidationError(
                "File must have an extension",
                field="filename",
                value=filename
            )
        
        # Check against allowed categories
        allowed_cats = allowed_categories or ['image', 'text', 'document']
        
        for category in allowed_cats:
            if file_ext in self.ALLOWED_EXTENSIONS.get(category, []):
                return category
        
        raise ValidationError(
            f"File extension '{file_ext}' is not allowed. "
            f"Allowed extensions: {[ext for cat in allowed_cats for ext in self.ALLOWED_EXTENSIONS.get(cat, [])]}",
            field="file_extension",
            value=file_ext
        )
    
    def validate_mime_type(self, file_path: str, expected_category: str) -> str:
        """Validate MIME type using python-magic.
        
        Args:
            file_path: Path to the file
            expected_category: Expected file category
            
        Returns:
            Detected MIME type
            
        Raises:
            ValidationError: If MIME type doesn't match extension or is dangerous
        """
        try:
            # Detect MIME type using magic
            mime_type = magic.from_file(file_path, mime=True)
            
            # Check if MIME type matches expected category
            allowed_mimes = self.ALLOWED_MIME_TYPES.get(expected_category, [])
            
            if mime_type not in allowed_mimes:
                raise ValidationError(
                    f"MIME type '{mime_type}' doesn't match expected category '{expected_category}'. "
                    f"Allowed MIME types: {allowed_mimes}",
                    field="mime_type",
                    value=mime_type
                )
            
            return mime_type
            
        except Exception as e:
            raise ValidationError(
                f"Could not determine file MIME type: {str(e)}",
                field="mime_type"
            )
    
    def check_file_signature(self, file_path: str) -> None:
        """Check file signature for dangerous content.
        
        Args:
            file_path: Path to the file
            
        Raises:
            ValidationError: If dangerous file signature is detected
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)  # Read first 16 bytes
            
            for signature in self.DANGEROUS_SIGNATURES:
                if header.startswith(signature):
                    raise ValidationError(
                        "File contains dangerous signature and cannot be processed",
                        field="file_signature"
                    )
                    
        except IOError as e:
            raise FileProcessingError(
                f"Could not read file for signature check: {str(e)}",
                processing_stage="signature_check"
            )
    
    def validate_image_file(self, file_path: str) -> Dict[str, any]:
        """Validate and extract metadata from image files.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with image metadata
            
        Raises:
            ValidationError: If image is invalid or corrupted
        """
        try:
            with Image.open(file_path) as img:
                # Check image dimensions
                width, height = img.size
                max_dims = self.config.analysis.image.max_dimensions
                
                if width > max_dims[0] or height > max_dims[1]:
                    raise ValidationError(
                        f"Image dimensions {width}x{height} exceed maximum allowed {max_dims[0]}x{max_dims[1]}",
                        field="image_dimensions",
                        value=f"{width}x{height}"
                    )
                
                # Extract metadata
                metadata = {
                    'width': width,
                    'height': height,
                    'format': img.format,
                    'mode': img.mode,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
                
                # Check for EXIF data (potential privacy concern)
                if hasattr(img, '_getexif') and img._getexif():
                    metadata['has_exif'] = True
                    logger.warning(f"Image contains EXIF data: {file_path}")
                else:
                    metadata['has_exif'] = False
                
                return metadata
                
        except Exception as e:
            raise ValidationError(
                f"Invalid or corrupted image file: {str(e)}",
                field="image_validation"
            )
    
    def validate_text_file(self, file_path: str) -> Dict[str, any]:
        """Validate and extract metadata from text files.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with text metadata
            
        Raises:
            ValidationError: If text file is invalid
        """
        try:
            # Try to read as UTF-8 first
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            content = None
            detected_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    detected_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValidationError(
                    "Could not decode text file with any supported encoding",
                    field="text_encoding"
                )
            
            # Check content length
            max_length = self.config.analysis.text.max_length
            if len(content) > max_length:
                raise ValidationError(
                    f"Text content length {len(content)} exceeds maximum {max_length} characters",
                    field="text_length",
                    value=len(content)
                )
            
            metadata = {
                'encoding': detected_encoding,
                'length': len(content),
                'lines': content.count('\n') + 1,
                'words': len(content.split()),
                'characters': len(content)
            }
            
            return metadata
            
        except Exception as e:
            raise ValidationError(
                f"Could not validate text file: {str(e)}",
                field="text_validation"
            )


class SecureFileManager:
    """Manages secure file storage and cleanup operations."""
    
    def __init__(self):
        """Initialize the secure file manager."""
        self.config = get_config()
        self.validator = FileValidator()
        
        # Setup directories
        self.upload_dir = Path(self.config.settings.upload_dir)
        self.temp_dir = Path(self.config.settings.temp_dir)
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist with proper permissions."""
        for directory in [self.upload_dir, self.temp_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Set restrictive permissions (owner read/write only)
                directory.chmod(0o700)
            except Exception as e:
                raise configuration_error(
                    f"Could not create directory {directory}: {str(e)}",
                    config_key="upload_dir" if directory == self.upload_dir else "temp_dir"
                )
    
    def generate_secure_filename(self, original_filename: str) -> str:
        """Generate a secure filename to prevent path traversal attacks.
        
        Args:
            original_filename: Original filename from upload
            
        Returns:
            Secure filename with timestamp and UUID
        """
        # Extract extension
        file_ext = Path(original_filename).suffix.lower()
        
        # Generate secure name with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        secure_name = f"{timestamp}_{unique_id}{file_ext}"
        
        # Additional sanitization
        secure_name = "".join(c for c in secure_name if c.isalnum() or c in '._-')
        
        return secure_name
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for integrity checking.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash as hexadecimal string
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            raise FileProcessingError(
                f"Could not calculate file hash: {str(e)}",
                processing_stage="hash_calculation"
            )
    
    def save_uploaded_file(
        self, 
        file_content: Union[BinaryIO, bytes], 
        original_filename: str,
        validate: bool = True
    ) -> Dict[str, any]:
        """Save uploaded file securely with validation.
        
        Args:
            file_content: File content as bytes or file-like object
            original_filename: Original filename
            validate: Whether to perform validation
            
        Returns:
            Dictionary with file information
            
        Raises:
            FileProcessingError: If file cannot be saved
            ValidationError: If validation fails
        """
        try:
            # Generate secure filename
            secure_filename = self.generate_secure_filename(original_filename)
            file_path = self.upload_dir / secure_filename
            
            # Write file content
            if isinstance(file_content, bytes):
                with open(file_path, 'wb') as f:
                    f.write(file_content)
            else:
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(file_content, f)
            
            # Get file stats
            file_stats = file_path.stat()
            file_size = file_stats.st_size
            
            # Validate file if requested
            file_info = {
                'original_filename': original_filename,
                'secure_filename': secure_filename,
                'file_path': str(file_path),
                'file_size': file_size,
                'upload_time': datetime.now(),
                'file_hash': self.calculate_file_hash(str(file_path))
            }
            
            if validate:
                # Basic validations
                self.validator.validate_file_size(file_size)
                category = self.validator.validate_file_extension(original_filename)
                mime_type = self.validator.validate_mime_type(str(file_path), category)
                self.validator.check_file_signature(str(file_path))
                
                # Category-specific validation
                if category == 'image':
                    image_metadata = self.validator.validate_image_file(str(file_path))
                    file_info.update(image_metadata)
                elif category == 'text':
                    text_metadata = self.validator.validate_text_file(str(file_path))
                    file_info.update(text_metadata)
                
                file_info.update({
                    'category': category,
                    'mime_type': mime_type,
                    'validated': True
                })
            
            logger.info(f"File saved successfully: {secure_filename}")
            return file_info
            
        except (ValidationError, FileProcessingError):
            # Clean up file if validation failed
            if file_path.exists():
                self._safe_delete_file(file_path)
            raise
        except Exception as e:
            # Clean up file on any error
            if file_path.exists():
                self._safe_delete_file(file_path)
            raise FileProcessingError(
                f"Could not save uploaded file: {str(e)}",
                processing_stage="file_save"
            )
    
    def create_temp_file(self, content: Union[str, bytes], suffix: str = '.tmp') -> str:
        """Create a temporary file with content.
        
        Args:
            content: Content to write to file
            suffix: File suffix/extension
            
        Returns:
            Path to the temporary file
        """
        try:
            # Create temporary file in our temp directory
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                dir=self.temp_dir,
                prefix=f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            )
            
            with os.fdopen(fd, 'wb' if isinstance(content, bytes) else 'w') as f:
                f.write(content)
            
            logger.debug(f"Created temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            raise FileProcessingError(
                f"Could not create temporary file: {str(e)}",
                processing_stage="temp_file_creation"
            )
    
    def _safe_delete_file(self, file_path: Union[str, Path]) -> bool:
        """Safely delete a file with error handling.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if file was deleted, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Could not delete file {file_path}: {str(e)}")
            return False
    
    def cleanup_file(self, file_path: str) -> bool:
        """Clean up a specific file.
        
        Args:
            file_path: Path to the file to clean up
            
        Returns:
            True if cleanup was successful
        """
        return self._safe_delete_file(file_path)
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up old files from upload and temp directories.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        for directory in [self.upload_dir, self.temp_dir]:
            try:
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            if self._safe_delete_file(file_path):
                                cleaned_count += 1
            except Exception as e:
                logger.error(f"Error during cleanup of {directory}: {str(e)}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old files")
        
        return cleaned_count
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileProcessingError(
                    f"File does not exist: {file_path}",
                    processing_stage="file_info"
                )
            
            stats = path.stat()
            
            return {
                'filename': path.name,
                'file_path': str(path),
                'file_size': stats.st_size,
                'created_time': datetime.fromtimestamp(stats.st_ctime),
                'modified_time': datetime.fromtimestamp(stats.st_mtime),
                'file_hash': self.calculate_file_hash(file_path)
            }
            
        except Exception as e:
            raise FileProcessingError(
                f"Could not get file info: {str(e)}",
                processing_stage="file_info"
            )


# Convenience functions for common operations

def validate_uploaded_file(
    file_content: Union[BinaryIO, bytes],
    filename: str,
    max_size: Optional[int] = None,
    allowed_categories: Optional[List[str]] = None
) -> Dict[str, any]:
    """Validate an uploaded file without saving it.
    
    Args:
        file_content: File content
        filename: Original filename
        max_size: Maximum file size in bytes
        allowed_categories: Allowed file categories
        
    Returns:
        Validation results
    """
    manager = SecureFileManager()
    
    # Create temporary file for validation
    temp_path = None
    try:
        if isinstance(file_content, bytes):
            content = file_content
        else:
            content = file_content.read()
            file_content.seek(0)  # Reset file pointer
        
        temp_path = manager.create_temp_file(content, suffix=Path(filename).suffix)
        
        # Validate
        manager.validator.validate_file_size(len(content), max_size)
        category = manager.validator.validate_file_extension(filename, allowed_categories)
        mime_type = manager.validator.validate_mime_type(temp_path, category)
        manager.validator.check_file_signature(temp_path)
        
        result = {
            'valid': True,
            'category': category,
            'mime_type': mime_type,
            'file_size': len(content)
        }
        
        # Add category-specific metadata
        if category == 'image':
            result.update(manager.validator.validate_image_file(temp_path))
        elif category == 'text':
            result.update(manager.validator.validate_text_file(temp_path))
        
        return result
        
    finally:
        if temp_path:
            manager.cleanup_file(temp_path)


def save_file_securely(
    file_content: Union[BinaryIO, bytes],
    filename: str,
    validate: bool = True
) -> Dict[str, any]:
    """Save a file securely with validation.
    
    Args:
        file_content: File content
        filename: Original filename
        validate: Whether to validate the file
        
    Returns:
        File information dictionary
    """
    manager = SecureFileManager()
    return manager.save_uploaded_file(file_content, filename, validate)


def cleanup_old_files(max_age_hours: int = 24) -> int:
    """Clean up old files.
    
    Args:
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of files cleaned up
    """
    manager = SecureFileManager()
    return manager.cleanup_old_files(max_age_hours)