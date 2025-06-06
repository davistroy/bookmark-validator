"""
CSV Field Validators

This module provides specific validators for each CSV field type in the raindrop.io
bookmark export/import format. It includes validators for URLs, dates, tags, titles,
and other CSV fields with appropriate validation logic.
"""

import re
from typing import Any, List, Optional
from urllib.parse import urlparse

from .input_validator import (
    Validator,
    ValidationResult,
    StringValidator,
    NumberValidator,
    DateTimeValidator,
    URLValidator,
    ListValidator,
    CompositeValidator,
    ValidationSeverity
)


class BookmarkIDValidator(StringValidator):
    """Validator for bookmark ID field"""
    
    def __init__(self):
        super().__init__(
            field_name="id",
            required=False,  # ID may be auto-generated
            allow_none=True,
            max_length=100,
            pattern=r'^[a-zA-Z0-9_-]*$'  # Allow alphanumeric, underscore, hyphen
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        # Additional validation for bookmark IDs
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            sanitized_id = result.sanitized_value.strip()
            
            # Check for common problematic patterns
            if sanitized_id.startswith('-') or sanitized_id.endswith('-'):
                result.add_warning("ID should not start or end with hyphen", self.field_name)
            
            if '__' in sanitized_id:
                result.add_warning("ID contains consecutive underscores", self.field_name)
            
            # Update sanitized value
            result.sanitized_value = sanitized_id
        
        return result


class BookmarkTitleValidator(StringValidator):
    """Validator for bookmark title field"""
    
    def __init__(self):
        super().__init__(
            field_name="title",
            required=False,  # Title can be extracted from content if missing
            allow_none=True,
            min_length=1,
            max_length=500  # Reasonable title length limit
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            title = result.sanitized_value
            
            # Check for overly long titles
            if len(title) > 200:
                result.add_warning(f"Title is quite long ({len(title)} chars)", self.field_name)
            
            # Check for suspicious content
            if title.count('\n') > 2:
                result.add_warning("Title contains multiple line breaks", self.field_name)
                # Clean up line breaks
                title = ' '.join(line.strip() for line in title.split('\n') if line.strip())
                result.sanitized_value = title
            
            # Check for HTML tags (should be cleaned)
            html_pattern = re.compile(r'<[^>]+>')
            if html_pattern.search(title):
                result.add_warning("Title contains HTML tags", self.field_name)
                # Remove HTML tags
                title = html_pattern.sub('', title).strip()
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                result.sanitized_value = title
        
        return result


class BookmarkNoteValidator(StringValidator):
    """Validator for bookmark note field"""
    
    def __init__(self):
        super().__init__(
            field_name="note",
            required=False,
            allow_none=True,
            max_length=2000  # Reasonable note length limit
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            note = result.sanitized_value
            
            # Check for very long notes
            if len(note) > 1000:
                result.add_warning(f"Note is quite long ({len(note)} chars)", self.field_name)
            
            # Clean up excessive whitespace while preserving structure
            # Replace multiple spaces with single space, but preserve line breaks
            note = re.sub(r'[ \t]+', ' ', note)  # Multiple spaces/tabs to single space
            note = re.sub(r'\n\s*\n\s*\n+', '\n\n', note)  # Multiple line breaks to double
            
            result.sanitized_value = note.strip()
        
        return result


class BookmarkExcerptValidator(StringValidator):
    """Validator for bookmark excerpt field"""
    
    def __init__(self):
        super().__init__(
            field_name="excerpt",
            required=False,
            allow_none=True,
            max_length=1000  # Excerpts should be concise
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            excerpt = result.sanitized_value
            
            # Check for very long excerpts
            if len(excerpt) > 500:
                result.add_warning(f"Excerpt is quite long ({len(excerpt)} chars)", self.field_name)
            
            # Clean up HTML if present
            html_pattern = re.compile(r'<[^>]+>')
            if html_pattern.search(excerpt):
                result.add_info("Removing HTML tags from excerpt", self.field_name)
                excerpt = html_pattern.sub('', excerpt)
                excerpt = re.sub(r'\s+', ' ', excerpt).strip()
                result.sanitized_value = excerpt
        
        return result


class BookmarkURLValidator(URLValidator):
    """Enhanced URL validator specifically for bookmarks"""
    
    def __init__(self):
        super().__init__(
            field_name="url",
            required=True,  # URLs are mandatory for bookmarks
            allow_none=False,
            allowed_schemes=['http', 'https'],
            security_check=True,
            normalize_url=True
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            url = result.sanitized_value
            
            # Additional bookmark-specific URL validation
            try:
                parsed = urlparse(url)
                
                # Check for localhost/development URLs
                if parsed.netloc.lower() in ['localhost', '127.0.0.1', '0.0.0.0']:
                    result.add_warning("URL points to localhost/development server", self.field_name)
                
                # Check for common development ports
                dev_ports = {3000, 3001, 8000, 8080, 8888, 9000}
                if parsed.port in dev_ports:
                    result.add_warning(f"URL uses common development port {parsed.port}", self.field_name)
                
                # Check for very long URLs
                if len(url) > 500:
                    result.add_warning(f"URL is very long ({len(url)} chars)", self.field_name)
                
                # Check for suspicious query parameters
                if parsed.query:
                    suspicious_params = ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']
                    query_lower = parsed.query.lower()
                    tracking_params = [param for param in suspicious_params if param in query_lower]
                    if tracking_params:
                        result.add_info(f"URL contains tracking parameters: {', '.join(tracking_params)}", self.field_name)
                
                # Check for fragments that might indicate specific sections
                if parsed.fragment and len(parsed.fragment) > 50:
                    result.add_info("URL has long fragment identifier", self.field_name)
                
            except Exception:
                pass  # URL parsing already handled by parent validator
        
        return result


class BookmarkFolderValidator(StringValidator):
    """Validator for bookmark folder field"""
    
    def __init__(self):
        super().__init__(
            field_name="folder",
            required=False,
            allow_none=True,
            max_length=500,
            pattern=r'^[^<>:"|?*\x00-\x1f\\]*$'  # Avoid filesystem-invalid characters
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            folder = result.sanitized_value.strip()
            
            # Clean up folder path
            # Replace backslashes with forward slashes
            folder = folder.replace('\\', '/')
            
            # Remove leading/trailing slashes
            folder = folder.strip('/')
            
            # Clean up multiple consecutive slashes
            folder = re.sub(r'/+', '/', folder)
            
            # Validate folder path components
            if folder:
                components = [comp.strip() for comp in folder.split('/') if comp.strip()]
                
                # Check individual folder component lengths
                for i, component in enumerate(components):
                    if len(component) > 100:
                        result.add_warning(f"Folder component '{component[:20]}...' is very long", self.field_name)
                    
                    # Check for problematic characters in components
                    if component.startswith('.') or component.endswith('.'):
                        result.add_warning(f"Folder component '{component}' starts/ends with period", self.field_name)
                
                # Check nesting depth
                if len(components) > 10:
                    result.add_warning(f"Folder path has deep nesting ({len(components)} levels)", self.field_name)
                
                # Reconstruct clean folder path
                folder = '/'.join(components)
            
            result.sanitized_value = folder
        
        return result


class BookmarkTagsValidator(Validator):
    """Validator for bookmark tags field (can be string or list)"""
    
    def __init__(self):
        super().__init__(
            field_name="tags",
            required=False,
            allow_none=True
        )
        self.max_tags = 20
        self.max_tag_length = 50
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        # Handle None/empty values
        if value is None or (isinstance(value, str) and not value.strip()):
            result.sanitized_value = []
            return result
        
        # Parse tags from various formats
        tags = []
        
        if isinstance(value, list):
            tags = [str(tag).strip() for tag in value if tag is not None]
        elif isinstance(value, str):
            tags_str = value.strip()
            
            # Handle quoted tag strings: "tag1, tag2, tag3"
            if tags_str.startswith('"') and tags_str.endswith('"'):
                tags_str = tags_str[1:-1]  # Remove quotes
            
            # Split on various delimiters
            if ',' in tags_str:
                tags = [tag.strip() for tag in tags_str.split(',')]
            elif ';' in tags_str:
                tags = [tag.strip() for tag in tags_str.split(';')]
            elif '|' in tags_str:
                tags = [tag.strip() for tag in tags_str.split('|')]
            else:
                # Single tag or space-separated
                if ' ' in tags_str and len(tags_str.split()) <= 5:
                    tags = tags_str.split()
                else:
                    tags = [tags_str]
        else:
            result.add_error(f"Tags must be string or list, got {type(value).__name__}", self.field_name)
            return result
        
        # Clean and validate individual tags
        cleaned_tags = []
        for i, tag in enumerate(tags):
            if not tag or not tag.strip():
                continue
            
            tag = tag.strip().lower()
            
            # Remove quotes and extra characters
            tag = tag.strip('"\'')
            
            # Validate tag format
            if len(tag) < 2:
                result.add_warning(f"Tag '{tag}' is too short (< 2 chars)", self.field_name)
                continue
            
            if len(tag) > self.max_tag_length:
                result.add_warning(f"Tag '{tag[:20]}...' is too long (> {self.max_tag_length} chars)", self.field_name)
                tag = tag[:self.max_tag_length]
            
            # Check for problematic characters
            if not re.match(r'^[a-zA-Z0-9\s\-_.]+$', tag):
                result.add_warning(f"Tag '{tag}' contains special characters", self.field_name)
                # Clean the tag
                tag = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', tag)
                if not tag or len(tag) < 2:
                    continue
            
            # Normalize whitespace in multi-word tags
            tag = re.sub(r'\s+', ' ', tag).strip()
            
            # Check for duplicates
            if tag not in cleaned_tags:
                cleaned_tags.append(tag)
            else:
                result.add_info(f"Removed duplicate tag '{tag}'", self.field_name)
        
        # Check total number of tags
        if len(cleaned_tags) > self.max_tags:
            result.add_warning(f"Too many tags ({len(cleaned_tags)} > {self.max_tags}), keeping first {self.max_tags}", self.field_name)
            cleaned_tags = cleaned_tags[:self.max_tags]
        
        result.sanitized_value = cleaned_tags
        return result


class BookmarkCreatedValidator(DateTimeValidator):
    """Validator for bookmark created date field"""
    
    def __init__(self):
        from datetime import datetime
        super().__init__(
            field_name="created",
            required=False,
            allow_none=True,
            min_date=datetime(1990, 1, 1),  # Reasonable minimum date
            max_date=None  # No max date (can be future)
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value:
            from datetime import datetime, timezone
            created_date = result.sanitized_value
            
            # Check if date is in the future
            now = datetime.now(timezone.utc)
            if created_date.tzinfo is None:
                # Assume UTC if no timezone info
                created_date = created_date.replace(tzinfo=timezone.utc)
            
            if created_date > now:
                result.add_warning("Created date is in the future", self.field_name)
            
            # Check for very old dates (might indicate incorrect parsing)
            if created_date.year < 1995:
                result.add_warning(f"Created date is very old ({created_date.year})", self.field_name)
        
        return result


class BookmarkCoverValidator(StringValidator):
    """Validator for bookmark cover field (usually URL to image)"""
    
    def __init__(self):
        super().__init__(
            field_name="cover",
            required=False,
            allow_none=True,
            max_length=1000
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            cover = result.sanitized_value.strip()
            
            # If it looks like a URL, validate it
            if cover and (cover.startswith('http://') or cover.startswith('https://')):
                try:
                    parsed = urlparse(cover)
                    
                    # Check if it looks like an image URL
                    path_lower = parsed.path.lower()
                    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
                    
                    if any(path_lower.endswith(ext) for ext in image_extensions):
                        result.add_info("Cover appears to be an image URL", self.field_name)
                    elif not path_lower or path_lower == '/':
                        result.add_warning("Cover URL points to root path", self.field_name)
                    
                    # Check for very long URLs
                    if len(cover) > 500:
                        result.add_warning(f"Cover URL is very long ({len(cover)} chars)", self.field_name)
                        
                except Exception:
                    result.add_warning("Cover field contains invalid URL", self.field_name)
            
            elif cover and not cover.startswith('http'):
                # Might be a relative path or other identifier
                result.add_info("Cover field does not appear to be a URL", self.field_name)
            
            result.sanitized_value = cover
        
        return result


class BookmarkHighlightsValidator(StringValidator):
    """Validator for bookmark highlights field"""
    
    def __init__(self):
        super().__init__(
            field_name="highlights",
            required=False,
            allow_none=True,
            max_length=5000  # Highlights can be lengthy
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = super().validate(value)
        
        if result.sanitized_value and isinstance(result.sanitized_value, str):
            highlights = result.sanitized_value
            
            # Check for very long highlights
            if len(highlights) > 2000:
                result.add_warning(f"Highlights field is very long ({len(highlights)} chars)", self.field_name)
            
            # Clean up excessive whitespace
            highlights = re.sub(r'\s+', ' ', highlights).strip()
            
            # Check for HTML content
            html_pattern = re.compile(r'<[^>]+>')
            if html_pattern.search(highlights):
                result.add_info("Highlights contain HTML tags", self.field_name)
                # Optionally clean HTML tags
                highlights = html_pattern.sub('', highlights)
                highlights = re.sub(r'\s+', ' ', highlights).strip()
            
            result.sanitized_value = highlights
        
        return result


class BookmarkFavoriteValidator(Validator):
    """Validator for bookmark favorite field (boolean)"""
    
    def __init__(self):
        super().__init__(
            field_name="favorite",
            required=False,
            allow_none=True
        )
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None or value == "":
            result.sanitized_value = False
            return result
        
        # Handle various boolean representations
        if isinstance(value, bool):
            result.sanitized_value = value
        elif isinstance(value, (int, float)):
            result.sanitized_value = bool(value)
        elif isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ('true', '1', 'yes', 'on', 'enabled', 'y'):
                result.sanitized_value = True
            elif value_lower in ('false', '0', 'no', 'off', 'disabled', 'n'):
                result.sanitized_value = False
            else:
                result.add_warning(f"Unclear boolean value '{value}', defaulting to False", self.field_name)
                result.sanitized_value = False
        else:
            result.add_warning(f"Cannot convert {type(value).__name__} to boolean, defaulting to False", self.field_name)
            result.sanitized_value = False
        
        return result


class BookmarkCompositeValidator(CompositeValidator):
    """Composite validator for complete bookmark records"""
    
    def __init__(self):
        # Create validators for all fields
        field_validators = {
            'id': BookmarkIDValidator(),
            'title': BookmarkTitleValidator(),
            'note': BookmarkNoteValidator(),
            'excerpt': BookmarkExcerptValidator(),
            'url': BookmarkURLValidator(),
            'folder': BookmarkFolderValidator(),
            'tags': BookmarkTagsValidator(),
            'created': BookmarkCreatedValidator(),
            'cover': BookmarkCoverValidator(),
            'highlights': BookmarkHighlightsValidator(),
            'favorite': BookmarkFavoriteValidator()
        }
        
        self.field_validators = field_validators
        super().__init__([], field_name="bookmark_record")
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate a complete bookmark record (dictionary)"""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, dict):
            result.add_critical("Bookmark record must be a dictionary", self.field_name)
            return result
        
        sanitized_record = {}
        
        # Validate each field
        for field_name, validator in self.field_validators.items():
            field_value = value.get(field_name)
            field_result = validator.validate(field_value)
            
            # Merge results
            result = result.merge(field_result)
            
            # Store sanitized value
            sanitized_record[field_name] = field_result.sanitized_value
        
        # Additional cross-field validation
        self._validate_cross_fields(sanitized_record, result)
        
        result.sanitized_value = sanitized_record
        return result
    
    def _validate_cross_fields(self, record: dict, result: ValidationResult) -> None:
        """Perform validation across multiple fields"""
        
        # Check if we have at least a URL
        if not record.get('url'):
            result.add_error("Bookmark must have a URL", self.field_name)
        
        # Check if we have at least a title or way to generate one
        if not record.get('title') and not record.get('url'):
            result.add_error("Bookmark must have either a title or URL", self.field_name)
        
        # Warn if bookmark has no descriptive content
        descriptive_fields = ['title', 'note', 'excerpt']
        has_description = any(record.get(field) for field in descriptive_fields)
        if not has_description:
            result.add_warning("Bookmark has no descriptive content (title, note, or excerpt)", self.field_name)
        
        # Check for consistency between URL and title
        url = record.get('url', '')
        title = record.get('title', '')
        if url and title:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                title_lower = title.lower()
                
                # Check if title is just the domain (might need better title)
                if domain and (domain in title_lower or title_lower in domain):
                    result.add_info("Title appears to be domain name - might benefit from content extraction", self.field_name)
                    
            except Exception:
                pass  # URL parsing already handled elsewhere


# Factory function to get appropriate validator for each field type
def get_field_validator(field_name: str) -> Validator:
    """
    Get the appropriate validator for a specific field name
    
    Args:
        field_name: Name of the field to validate
        
    Returns:
        Appropriate Validator instance
    """
    validators = {
        'id': BookmarkIDValidator(),
        'title': BookmarkTitleValidator(),
        'note': BookmarkNoteValidator(),
        'excerpt': BookmarkExcerptValidator(),
        'url': BookmarkURLValidator(),
        'folder': BookmarkFolderValidator(),
        'tags': BookmarkTagsValidator(),
        'created': BookmarkCreatedValidator(),
        'cover': BookmarkCoverValidator(),
        'highlights': BookmarkHighlightsValidator(),
        'favorite': BookmarkFavoriteValidator()
    }
    
    return validators.get(field_name, StringValidator(field_name=field_name))


def validate_bookmark_record(record: dict) -> ValidationResult:
    """
    Validate a complete bookmark record
    
    Args:
        record: Dictionary containing bookmark data
        
    Returns:
        ValidationResult with validation details
    """
    validator = BookmarkCompositeValidator()
    return validator.validate(record)


def validate_csv_row(row_data: dict, expected_columns: List[str]) -> ValidationResult:
    """
    Validate a CSV row against expected column structure
    
    Args:
        row_data: Dictionary from CSV row
        expected_columns: List of expected column names
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(is_valid=True)
    
    # Check for missing columns
    missing_columns = set(expected_columns) - set(row_data.keys())
    if missing_columns:
        result.add_error(f"Missing columns: {', '.join(missing_columns)}")
    
    # Check for extra columns
    extra_columns = set(row_data.keys()) - set(expected_columns)
    if extra_columns:
        result.add_warning(f"Extra columns: {', '.join(extra_columns)}")
    
    # Validate each field that we have a validator for
    sanitized_row = {}
    for column in expected_columns:
        if column in row_data:
            validator = get_field_validator(column)
            field_result = validator.validate(row_data[column])
            result = result.merge(field_result)
            sanitized_row[column] = field_result.sanitized_value
        else:
            sanitized_row[column] = None
    
    result.sanitized_value = sanitized_row
    return result