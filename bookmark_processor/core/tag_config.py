"""
Tag Configuration Module

Provides configurable tag settings including protected tags, synonym mappings,
tag hierarchy, and vocabulary customization. Supports TOML configuration files.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tomllib
except ImportError:
    # Python < 3.11 fallback
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class TagConfig:
    """User-configurable tag settings."""

    # Protected tags (never consolidated or modified)
    protected_tags: Set[str] = field(default_factory=lambda: {
        "important", "to-read", "reference", "archived", "favorite"
    })

    # Synonym mappings (key -> normalized form)
    synonyms: Dict[str, str] = field(default_factory=lambda: {
        "artificial-intelligence": "ai",
        "machine-learning": "ml",
        "js": "javascript",
        "py": "python",
        "ts": "typescript",
        "ui-ux": "design",
        "user-interface": "ui",
        "user-experience": "ux",
        "dev": "development",
        "prog": "programming",
    })

    # Hierarchy definitions (tag -> parent/tag path)
    hierarchy: Dict[str, str] = field(default_factory=dict)

    # Target counts
    target_unique_tags: int = 150
    max_tags_per_bookmark: int = 5
    min_tag_frequency: int = 2

    # Quality thresholds
    quality_threshold: float = 0.3
    confidence_threshold: float = 0.5

    # Category mappings for hierarchical tags
    category_mappings: Dict[str, List[str]] = field(default_factory=lambda: {
        "technology": ["programming", "development", "software", "code", "devops"],
        "technology/ai": ["ai", "ml", "machine-learning", "deep-learning", "neural"],
        "technology/web": ["web", "frontend", "backend", "javascript", "html", "css"],
        "technology/mobile": ["mobile", "android", "ios", "flutter", "react-native"],
        "design": ["design", "ui", "ux", "graphic", "typography"],
        "business": ["business", "startup", "marketing", "finance"],
        "education": ["tutorial", "course", "learning", "guide", "documentation"],
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "protected_tags": list(self.protected_tags),
            "synonyms": self.synonyms,
            "hierarchy": self.hierarchy,
            "target_unique_tags": self.target_unique_tags,
            "max_tags_per_bookmark": self.max_tags_per_bookmark,
            "min_tag_frequency": self.min_tag_frequency,
            "quality_threshold": self.quality_threshold,
            "confidence_threshold": self.confidence_threshold,
            "category_mappings": self.category_mappings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TagConfig":
        """Create from dictionary."""
        protected = data.get("protected_tags", [])
        if isinstance(protected, list):
            protected = set(protected)

        return cls(
            protected_tags=protected,
            synonyms=data.get("synonyms", {}),
            hierarchy=data.get("hierarchy", {}),
            target_unique_tags=data.get("target_unique_tags", 150),
            max_tags_per_bookmark=data.get("max_tags_per_bookmark", 5),
            min_tag_frequency=data.get("min_tag_frequency", 2),
            quality_threshold=data.get("quality_threshold", 0.3),
            confidence_threshold=data.get("confidence_threshold", 0.5),
            category_mappings=data.get("category_mappings", {}),
        )

    @classmethod
    def from_toml_file(cls, file_path: str) -> "TagConfig":
        """
        Load configuration from a TOML file.

        Args:
            file_path: Path to TOML configuration file

        Returns:
            TagConfig instance

        Raises:
            ValueError: If TOML parsing is not available
            FileNotFoundError: If file doesn't exist
        """
        if tomllib is None:
            raise ValueError(
                "TOML parsing not available. Install tomli for Python < 3.11 or use Python 3.11+"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        logger = logging.getLogger(__name__)
        logger.info(f"Loading tag configuration from: {file_path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Extract tags section
        tags_section = data.get("tags", {})

        # Build config from TOML structure
        config_data = {}

        # Protected tags
        if "protected_tags" in tags_section:
            config_data["protected_tags"] = set(tags_section["protected_tags"])

        # Synonyms (nested table in TOML)
        if "synonyms" in tags_section:
            config_data["synonyms"] = tags_section["synonyms"]

        # Hierarchy
        if "hierarchy" in tags_section:
            config_data["hierarchy"] = tags_section["hierarchy"]

        # Numeric settings
        for key in ["target_unique_tags", "max_tags_per_bookmark", "min_tag_frequency"]:
            if key in tags_section:
                config_data[key] = tags_section[key]

        # Float settings
        for key in ["quality_threshold", "confidence_threshold"]:
            if key in tags_section:
                config_data[key] = tags_section[key]

        # Category mappings
        if "category_mappings" in tags_section:
            config_data["category_mappings"] = tags_section["category_mappings"]

        logger.info(f"Loaded tag config with {len(config_data.get('protected_tags', []))} protected tags")

        return cls.from_dict(config_data)

    def save_to_toml(self, file_path: str) -> None:
        """
        Save configuration to a TOML file.

        Args:
            file_path: Path to save TOML file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build TOML content
        lines = ["[tags]"]
        lines.append(f"protected_tags = {list(self.protected_tags)}")
        lines.append(f"target_unique_tags = {self.target_unique_tags}")
        lines.append(f"max_tags_per_bookmark = {self.max_tags_per_bookmark}")
        lines.append(f"min_tag_frequency = {self.min_tag_frequency}")
        lines.append(f"quality_threshold = {self.quality_threshold}")
        lines.append(f"confidence_threshold = {self.confidence_threshold}")
        lines.append("")

        # Synonyms section
        if self.synonyms:
            lines.append("[tags.synonyms]")
            for key, value in sorted(self.synonyms.items()):
                lines.append(f'"{key}" = "{value}"')
            lines.append("")

        # Hierarchy section
        if self.hierarchy:
            lines.append("[tags.hierarchy]")
            for key, value in sorted(self.hierarchy.items()):
                lines.append(f'"{key}" = "{value}"')
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logging.getLogger(__name__).info(f"Saved tag configuration to: {file_path}")


@dataclass
class TagWithConfidence:
    """A tag with its confidence score."""

    tag: str
    confidence: float
    source: str = "extracted"  # extracted, existing, ai_generated, hierarchy

    def to_tuple(self) -> Tuple[str, float]:
        """Convert to simple tuple."""
        return (self.tag, self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag": self.tag,
            "confidence": self.confidence,
            "source": self.source,
        }


class TagNormalizer:
    """Normalizes tags using configuration."""

    def __init__(self, config: Optional[TagConfig] = None):
        """
        Initialize tag normalizer.

        Args:
            config: Tag configuration
        """
        self.config = config or TagConfig()
        self.logger = logging.getLogger(__name__)

    def normalize_tag(self, tag: str) -> str:
        """
        Normalize a tag by applying synonyms and cleaning.

        Args:
            tag: Raw tag string

        Returns:
            Normalized tag
        """
        # Basic cleaning
        normalized = tag.strip().lower()
        normalized = normalized.replace("_", "-")

        # Skip protected tags
        if normalized in self.config.protected_tags:
            return normalized

        # Apply synonym mapping
        if normalized in self.config.synonyms:
            normalized = self.config.synonyms[normalized]

        return normalized

    def apply_hierarchy(self, tag: str) -> str:
        """
        Apply hierarchy transformation to a tag.

        Args:
            tag: Tag to transform

        Returns:
            Tag with hierarchy applied (e.g., "ai" -> "technology/ai")
        """
        normalized = self.normalize_tag(tag)

        if normalized in self.config.hierarchy:
            return self.config.hierarchy[normalized]

        # Check category mappings
        for category, tags in self.config.category_mappings.items():
            if normalized in tags:
                return f"{category}/{normalized}" if "/" not in category else category

        return normalized

    def is_protected(self, tag: str) -> bool:
        """
        Check if a tag is protected.

        Args:
            tag: Tag to check

        Returns:
            True if tag is protected
        """
        return tag.strip().lower() in self.config.protected_tags

    def normalize_tags(self, tags: List[str]) -> List[str]:
        """
        Normalize a list of tags.

        Args:
            tags: List of tags to normalize

        Returns:
            List of normalized tags (deduplicated)
        """
        seen = set()
        result = []

        for tag in tags:
            normalized = self.normalize_tag(tag)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return result

    def normalize_tags_with_confidence(
        self,
        tags_with_confidence: List[Tuple[str, float]],
    ) -> List[TagWithConfidence]:
        """
        Normalize tags while preserving confidence scores.

        Args:
            tags_with_confidence: List of (tag, confidence) tuples

        Returns:
            List of TagWithConfidence objects
        """
        seen = {}
        result = []

        for tag, confidence in tags_with_confidence:
            normalized = self.normalize_tag(tag)
            if not normalized:
                continue

            # Keep highest confidence for duplicates
            if normalized in seen:
                if confidence > seen[normalized].confidence:
                    seen[normalized].confidence = confidence
            else:
                twc = TagWithConfidence(
                    tag=normalized,
                    confidence=confidence,
                    source="extracted",
                )
                seen[normalized] = twc
                result.append(twc)

        return sorted(result, key=lambda x: x.confidence, reverse=True)

    def get_category_for_tag(self, tag: str) -> Optional[str]:
        """
        Get the category for a tag based on category mappings.

        Args:
            tag: Tag to categorize

        Returns:
            Category name or None
        """
        normalized = self.normalize_tag(tag)

        for category, tags in self.config.category_mappings.items():
            if normalized in tags:
                return category.split("/")[0]  # Return top-level category

        return None
