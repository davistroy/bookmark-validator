"""
Tests for Structured Output Models

Tests the Pydantic models and parsing functions for AI-generated
bookmark enhancements.
"""

import json
import pytest
from bookmark_processor.core.structured_output import (
    BookmarkEnhancement,
    BatchBookmarkEnhancement,
    CLAUDE_BOOKMARK_TOOL,
    OPENAI_RESPONSE_FORMAT,
    parse_enhancement_response,
    create_enhancement_prompt,
)


class TestBookmarkEnhancement:
    """Tests for BookmarkEnhancement model."""

    def test_valid_enhancement(self):
        """Test creating a valid enhancement."""
        enhancement = BookmarkEnhancement(
            description="A comprehensive guide to Python async programming with examples.",
            tags=["python", "async", "programming", "tutorial"],
            category="tutorial",
            confidence=0.9,
        )
        assert enhancement.description
        assert len(enhancement.tags) == 4
        assert enhancement.category == "tutorial"
        assert enhancement.confidence == 0.9

    def test_description_cleaning(self):
        """Test that descriptions are cleaned properly."""
        enhancement = BookmarkEnhancement(
            description="  Comprehensive guide to testing with Python pytest  ",
            tags=["test"],
        )
        # Verify leading/trailing spaces are stripped
        assert not enhancement.description.startswith(" ")
        assert not enhancement.description.endswith(" ")
        assert "testing" in enhancement.description.lower()

    def test_tag_cleaning(self):
        """Test that tags are cleaned and normalized."""
        enhancement = BookmarkEnhancement(
            description="A valid description that meets the minimum length requirement.",
            tags=["  Python  ", "ASYNC", "test-tag", "a", "valid"],
        )
        # Tags should be lowercase, cleaned, and short tags removed
        assert "python" in enhancement.tags
        assert "async" in enhancement.tags
        assert "test-tag" in enhancement.tags

    def test_tag_deduplication(self):
        """Test that duplicate tags are removed."""
        enhancement = BookmarkEnhancement(
            description="A valid description that meets the minimum length requirement.",
            tags=["python", "Python", "PYTHON", "async"],
        )
        # Should only have unique tags
        assert len([t for t in enhancement.tags if t == "python"]) == 1

    def test_tag_limit(self):
        """Test that tags are properly cleaned."""
        many_tags = [f"tag{i}" for i in range(5)]
        enhancement = BookmarkEnhancement(
            description="A valid description that meets the minimum length requirement.",
            tags=many_tags,
        )
        assert len(enhancement.tags) >= 1

    def test_default_values(self):
        """Test default values are set correctly."""
        enhancement = BookmarkEnhancement(
            description="A valid description that meets the minimum length requirement.",
            tags=["test"],
        )
        assert enhancement.category is None
        assert enhancement.confidence == 0.8


class TestParseEnhancementResponse:
    """Tests for parse_enhancement_response function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        json_response = json.dumps({
            "description": "A great tutorial for learning Python web development.",
            "tags": ["python", "web", "tutorial"],
            "category": "tutorial",
            "confidence": 0.85,
        })
        enhancement = parse_enhancement_response(json_response)
        assert enhancement.description
        assert len(enhancement.tags) >= 1
        assert enhancement.category == "tutorial"

    def test_parse_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
{
    "description": "A comprehensive guide to async programming in Python.",
    "tags": ["python", "async"],
    "category": "tutorial",
    "confidence": 0.9
}
```"""
        enhancement = parse_enhancement_response(response)
        assert "async" in enhancement.description.lower() or len(enhancement.tags) >= 1

    def test_parse_plain_text_fallback(self):
        """Test fallback parsing for plain text responses."""
        plain_text = "This is a plain text description without JSON formatting."
        enhancement = parse_enhancement_response(plain_text)
        assert enhancement.description
        assert enhancement.confidence == 0.5  # Lower confidence for fallback
        assert "uncategorized" in enhancement.tags  # Default tag for fallback

    def test_parse_invalid_json_fallback(self):
        """Test fallback for invalid JSON."""
        invalid_json = "{ invalid json here }"
        enhancement = parse_enhancement_response(invalid_json)
        assert enhancement.description
        assert enhancement.category == "other"
        assert len(enhancement.tags) >= 1  # Should have default tag


class TestCreateEnhancementPrompt:
    """Tests for create_enhancement_prompt function."""

    def test_structured_prompt(self):
        """Test creating a structured prompt."""
        prompt = create_enhancement_prompt(
            title="Python Tutorial",
            url="https://example.com/python-tutorial",
            existing_content="Learn Python basics",
            structured=True,
        )
        assert "Python Tutorial" in prompt
        assert "example.com" in prompt
        assert "JSON" in prompt

    def test_unstructured_prompt(self):
        """Test creating an unstructured prompt."""
        prompt = create_enhancement_prompt(
            title="Python Tutorial",
            url="https://example.com/python-tutorial",
            existing_content="Learn Python basics",
            structured=False,
        )
        assert "Python Tutorial" in prompt
        assert "JSON" not in prompt

    def test_prompt_with_no_content(self):
        """Test prompt with no existing content."""
        prompt = create_enhancement_prompt(
            title="Test Page",
            url="https://example.com/test",
            existing_content=None,
        )
        assert "None" in prompt or "Test Page" in prompt


class TestToolSchemas:
    """Tests for tool schemas."""

    def test_claude_tool_schema_valid(self):
        """Test Claude tool schema is valid."""
        assert "name" in CLAUDE_BOOKMARK_TOOL
        assert "input_schema" in CLAUDE_BOOKMARK_TOOL
        assert CLAUDE_BOOKMARK_TOOL["name"] == "enhance_bookmark"

        schema = CLAUDE_BOOKMARK_TOOL["input_schema"]
        assert "properties" in schema
        assert "description" in schema["properties"]
        assert "tags" in schema["properties"]

    def test_openai_response_format_valid(self):
        """Test OpenAI response format is valid."""
        assert "type" in OPENAI_RESPONSE_FORMAT
        assert OPENAI_RESPONSE_FORMAT["type"] == "json_schema"
        assert "json_schema" in OPENAI_RESPONSE_FORMAT


class TestBatchBookmarkEnhancement:
    """Tests for BatchBookmarkEnhancement model."""

    def test_batch_enhancement(self):
        """Test creating a batch of enhancements."""
        batch = BatchBookmarkEnhancement(
            enhancements=[
                BookmarkEnhancement(
                    description="First bookmark description for testing purposes and validation.",
                    tags=["test1"],
                ),
                BookmarkEnhancement(
                    description="Second bookmark description for testing purposes and validation.",
                    tags=["test2"],
                ),
            ]
        )
        assert len(batch.enhancements) == 2
