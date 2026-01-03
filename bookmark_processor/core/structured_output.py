"""
Structured Output Models for AI Responses

This module defines Pydantic models for structured output from AI providers,
ensuring reliable and type-safe responses for bookmark enhancement.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BookmarkEnhancement(BaseModel):
    """Structured output for bookmark enhancement from AI."""

    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Enhanced description of the bookmark (100-150 chars ideal)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Relevant tags for the bookmark",
    )
    category: Optional[str] = Field(
        default=None,
        description="Primary category for the bookmark",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for the enhancement",
    )

    @field_validator("description")
    @classmethod
    def clean_description(cls, v: str) -> str:
        """Clean and validate description."""
        # Remove leading/trailing whitespace
        v = v.strip()
        # Remove common AI preambles
        prefixes_to_remove = [
            "This bookmark is about",
            "This is a",
            "This website",
            "Here is",
        ]
        for prefix in prefixes_to_remove:
            if v.lower().startswith(prefix.lower()):
                v = v[len(prefix) :].strip()
                # Capitalize first letter
                if v:
                    v = v[0].upper() + v[1:]
        return v

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: List[str]) -> List[str]:
        """Clean and validate tags."""
        cleaned = []
        for tag in v:
            # Normalize tag
            tag = tag.strip().lower()
            # Remove special characters except hyphens
            tag = "".join(c for c in tag if c.isalnum() or c == "-")
            # Skip empty or too short tags
            if len(tag) >= 2 and tag not in cleaned:
                cleaned.append(tag)
        return cleaned[:10]  # Limit to 10 tags


class BatchBookmarkEnhancement(BaseModel):
    """Structured output for batch bookmark enhancement."""

    enhancements: List[BookmarkEnhancement] = Field(
        ...,
        description="List of bookmark enhancements in order",
    )


# JSON Schema for Claude tool use
CLAUDE_BOOKMARK_TOOL = {
    "name": "enhance_bookmark",
    "description": "Generate an enhanced description and tags for a bookmark",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Enhanced description (100-150 characters)",
                "minLength": 50,
                "maxLength": 200,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant tags (3-7 tags)",
                "minItems": 1,
                "maxItems": 10,
            },
            "category": {
                "type": "string",
                "description": "Primary category",
                "enum": [
                    "tutorial",
                    "documentation",
                    "article",
                    "tool",
                    "video",
                    "code",
                    "research",
                    "resource",
                    "news",
                    "reference",
                    "other",
                ],
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score (0-1)",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["description", "tags"],
    },
}


# OpenAI JSON Schema response format
OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "bookmark_enhancement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Enhanced description (100-150 characters)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant tags (3-7 tags)",
                },
                "category": {
                    "type": "string",
                    "description": "Primary category",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score (0-1)",
                },
            },
            "required": ["description", "tags", "category", "confidence"],
            "additionalProperties": False,
        },
    },
}


def parse_enhancement_response(response_text: str) -> BookmarkEnhancement:
    """
    Parse AI response text into structured BookmarkEnhancement.

    Handles both JSON responses and text responses with fallback parsing.

    Args:
        response_text: Raw response text from AI

    Returns:
        BookmarkEnhancement object
    """
    import json
    import re

    # Try to parse as JSON first
    try:
        # Handle potential markdown code blocks
        if "```json" in response_text:
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        elif "```" in response_text:
            json_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

        data = json.loads(response_text.strip())
        return BookmarkEnhancement(**data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: parse as plain text
    # Use the entire response as description
    description = response_text.strip()

    # Truncate if too long
    if len(description) > 500:
        description = description[:497] + "..."

    # Ensure minimum length
    if len(description) < 10:
        description = "No description available"

    return BookmarkEnhancement(
        description=description,
        tags=["uncategorized"],  # Default tag for fallback
        category="other",
        confidence=0.5,
    )


def create_enhancement_prompt(
    title: str,
    url: str,
    existing_content: Optional[str] = None,
    structured: bool = True,
) -> str:
    """
    Create a prompt for bookmark enhancement.

    Args:
        title: Bookmark title
        url: Bookmark URL
        existing_content: Existing note/excerpt content
        structured: Whether to request structured JSON output

    Returns:
        Formatted prompt string
    """
    from urllib.parse import urlparse

    try:
        domain = urlparse(url).netloc or "unknown"
    except Exception:
        domain = "unknown"

    base_prompt = f"""Analyze this bookmark and provide an enhanced description with relevant tags.

Title: {title}
Domain: {domain}
Existing Content: {existing_content or 'None'}

Requirements:
- Description: 100-150 characters, actionable and specific
- Tags: 3-7 relevant keywords
- Focus on what problem this solves or what users can learn/do
- Avoid generic phrases like "This is a website about..."
"""

    if structured:
        base_prompt += """
Respond with a JSON object:
{
  "description": "Your enhanced description here",
  "tags": ["tag1", "tag2", "tag3"],
  "category": "one of: article, tutorial, documentation, tool, video, code, etc.",
  "confidence": 0.8
}"""
    else:
        base_prompt += "\nProvide just the enhanced description."

    return base_prompt
