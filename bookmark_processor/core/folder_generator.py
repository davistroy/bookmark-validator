"""
AI-Powered Folder Structure Generator

Generates semantic folder structures for bookmarks based on content analysis,
with support for hierarchical organization and folder size limits.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ai_processor import AIProcessingResult
from .content_analyzer import ContentData
from .data_models import Bookmark


@dataclass
class FolderNode:
    """Represents a folder in the hierarchy"""

    name: str
    path: str
    parent: Optional["FolderNode"] = None
    children: List["FolderNode"] = None
    bookmarks: List[Bookmark] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.bookmarks is None:
            self.bookmarks = []

    def add_bookmark(self, bookmark: Bookmark) -> None:
        """Add a bookmark to this folder"""
        self.bookmarks.append(bookmark)

    def add_child(self, child: "FolderNode") -> None:
        """Add a child folder"""
        child.parent = self
        self.children.append(child)

    def get_bookmark_count(self) -> int:
        """Get total bookmark count including children"""
        count = len(self.bookmarks)
        for child in self.children:
            count += child.get_bookmark_count()
        return count

    def get_full_path(self) -> str:
        """Get the full path from root"""
        if self.parent and self.parent.name != "root":
            return f"{self.parent.get_full_path()}/{self.name}"
        return self.name


@dataclass
class FolderGenerationResult:
    """Results from folder generation"""

    root_folder: FolderNode
    folder_assignments: Dict[str, str]  # url -> folder path
    total_folders: int
    max_depth: int
    folder_stats: Dict[str, int]  # folder path -> bookmark count
    processing_time: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "folder_assignments": self.folder_assignments,
            "total_folders": self.total_folders,
            "max_depth": self.max_depth,
            "folder_stats": self.folder_stats,
            "processing_time": self.processing_time,
        }


class AIFolderGenerator:
    """
    Generates semantic folder structures using AI analysis.

    Features:
    - Content-based semantic analysis
    - Hierarchical folder organization
    - 20-bookmark per folder limit with sub-folder creation
    - Original folder hints for context
    - AI-powered categorization
    """

    def __init__(
        self,
        max_bookmarks_per_folder: int = 20,
        ai_engine: str = "local",
        api_key: Optional[str] = None,
    ):
        """
        Initialize folder generator.

        Args:
            max_bookmarks_per_folder: Maximum bookmarks allowed per folder
            ai_engine: AI engine to use ("local", "claude", "openai")
            api_key: API key for cloud AI services
        """
        self.max_bookmarks_per_folder = max_bookmarks_per_folder

        # For now, we'll just use local AI processor since folder generation
        # doesn't need cloud AI
        # This can be enhanced later to use AIFactory properly
        from .ai_processor import EnhancedAIProcessor

        self.ai_processor = EnhancedAIProcessor()

        # Common technology and topic patterns
        self._init_category_patterns()

        logging.info(f"Initialized AI folder generator with {ai_engine} engine")

    def _init_category_patterns(self) -> None:
        """Initialize category detection patterns"""
        self.category_patterns = {
            "Development": {
                "keywords": [
                    "programming",
                    "coding",
                    "developer",
                    "software",
                    "github",
                    "api",
                    "framework",
                ],
                "domains": ["github.com", "stackoverflow.com", "dev.to"],
                "subcategories": {
                    "Frontend": [
                        "react",
                        "vue",
                        "angular",
                        "css",
                        "html",
                        "javascript",
                        "typescript",
                    ],
                    "Backend": [
                        "python",
                        "java",
                        "node",
                        "django",
                        "flask",
                        "api",
                        "database",
                    ],
                    "DevOps": [
                        "docker",
                        "kubernetes",
                        "aws",
                        "azure",
                        "ci/cd",
                        "jenkins",
                    ],
                    "Mobile": ["android", "ios", "react native", "flutter", "swift"],
                },
            },
            "AI & Machine Learning": {
                "keywords": [
                    "ai",
                    "artificial intelligence",
                    "machine learning",
                    "deep learning",
                    "neural",
                    "model",
                ],
                "domains": ["arxiv.org", "paperswithcode.com", "huggingface.co"],
                "subcategories": {
                    "Research": ["paper", "research", "study", "arxiv"],
                    "Tools": ["tensorflow", "pytorch", "scikit", "jupyter"],
                    "Applications": ["nlp", "computer vision", "robotics", "chatbot"],
                },
            },
            "Design": {
                "keywords": [
                    "design",
                    "ux",
                    "ui",
                    "graphic",
                    "typography",
                    "color",
                    "layout",
                ],
                "domains": ["dribbble.com", "behance.net", "figma.com"],
                "subcategories": {
                    "UI/UX": [
                        "user interface",
                        "user experience",
                        "wireframe",
                        "prototype",
                    ],
                    "Graphic Design": ["logo", "branding", "illustration", "poster"],
                    "Web Design": ["responsive", "layout", "css", "animation"],
                },
            },
            "Business": {
                "keywords": [
                    "business",
                    "startup",
                    "entrepreneur",
                    "marketing",
                    "finance",
                    "strategy",
                ],
                "domains": ["forbes.com", "bloomberg.com", "businessinsider.com"],
                "subcategories": {
                    "Startups": ["startup", "founder", "venture", "funding"],
                    "Marketing": ["seo", "social media", "content", "advertising"],
                    "Finance": ["investment", "stock", "crypto", "banking"],
                },
            },
            "Education": {
                "keywords": [
                    "tutorial",
                    "course",
                    "learn",
                    "education",
                    "training",
                    "guide",
                ],
                "domains": ["coursera.org", "udemy.com", "edx.org", "khanacademy.org"],
                "subcategories": {
                    "Online Courses": ["course", "mooc", "certification"],
                    "Tutorials": ["tutorial", "how-to", "guide", "walkthrough"],
                    "Documentation": ["docs", "reference", "manual", "api"],
                },
            },
            "News & Media": {
                "keywords": ["news", "article", "blog", "media", "journalism"],
                "domains": ["medium.com", "nytimes.com", "bbc.com", "reddit.com"],
                "subcategories": {
                    "Tech News": ["tech", "technology", "gadget", "innovation"],
                    "General News": ["politics", "world", "breaking", "current"],
                    "Blogs": ["blog", "personal", "opinion", "story"],
                },
            },
            "Tools & Resources": {
                "keywords": [
                    "tool",
                    "utility",
                    "resource",
                    "generator",
                    "converter",
                    "calculator",
                ],
                "domains": ["codepen.io", "jsfiddle.net", "regex101.com"],
                "subcategories": {
                    "Dev Tools": ["compiler", "debugger", "linter", "formatter"],
                    "Productivity": ["todo", "calendar", "notes", "workflow"],
                    "Utilities": ["converter", "generator", "calculator", "checker"],
                },
            },
            "Reference": {
                "keywords": [
                    "documentation",
                    "reference",
                    "wiki",
                    "encyclopedia",
                    "dictionary",
                ],
                "domains": [
                    "wikipedia.org",
                    "docs.python.org",
                    "developer.mozilla.org",
                ],
                "subcategories": {
                    "Documentation": ["api", "docs", "reference", "manual"],
                    "Wiki": ["wikipedia", "wiki", "encyclopedia"],
                    "Standards": ["rfc", "specification", "standard", "protocol"],
                },
            },
        }

    def generate_folder_structure(
        self,
        bookmarks: List[Bookmark],
        content_data_map: Optional[Dict[str, ContentData]] = None,
        ai_results_map: Optional[Dict[str, AIProcessingResult]] = None,
        original_folders_map: Optional[Dict[str, str]] = None,
    ) -> FolderGenerationResult:
        """
        Generate AI-powered folder structure for bookmarks.

        Args:
            bookmarks: List of bookmarks to organize
            content_data_map: Content analysis results
            ai_results_map: AI processing results
            original_folders_map: Original folder paths as hints (url -> folder)

        Returns:
            FolderGenerationResult with folder assignments
        """
        import time

        start_time = time.time()

        logging.info(f"Generating folder structure for {len(bookmarks)} bookmarks")

        # Initialize data maps if not provided
        if content_data_map is None:
            content_data_map = {}
        if ai_results_map is None:
            ai_results_map = {}
        if original_folders_map is None:
            original_folders_map = {}

        # Create root folder
        root = FolderNode(name="root", path="")
        folder_assignments = {}

        # Step 1: Analyze bookmarks and assign initial categories
        categorized_bookmarks = self._categorize_bookmarks(
            bookmarks, content_data_map, ai_results_map, original_folders_map
        )

        # Step 2: Build folder hierarchy
        folder_tree = self._build_folder_hierarchy(categorized_bookmarks, root)

        # Step 3: Apply size limits and create sub-folders
        self._apply_folder_limits(folder_tree)

        # Step 4: Assign bookmarks to final folders
        folder_assignments = self._assign_bookmarks_to_folders(folder_tree)

        # Calculate statistics
        folder_stats = self._calculate_folder_stats(folder_tree)
        max_depth = self._calculate_max_depth(folder_tree)

        processing_time = time.time() - start_time

        result = FolderGenerationResult(
            root_folder=folder_tree,
            folder_assignments=folder_assignments,
            total_folders=len(folder_stats),
            max_depth=max_depth,
            folder_stats=folder_stats,
            processing_time=processing_time,
        )

        logging.info(
            f"Generated {result.total_folders} folders with max depth "
            f"{result.max_depth}"
        )

        return result

    def _categorize_bookmarks(
        self,
        bookmarks: List[Bookmark],
        content_data_map: Dict[str, ContentData],
        ai_results_map: Dict[str, AIProcessingResult],
        original_folders_map: Dict[str, str],
    ) -> Dict[str, List[Tuple[Bookmark, str]]]:
        """
        Categorize bookmarks based on content analysis.

        Returns:
            Dict mapping category -> list of (bookmark, subcategory)
        """
        categorized = defaultdict(list)

        for bookmark in bookmarks:
            # Get all available data
            content = content_data_map.get(bookmark.url)
            ai_result = ai_results_map.get(bookmark.url)
            original_folder = original_folders_map.get(bookmark.url, "")

            # Determine category and subcategory
            category, subcategory = self._determine_category(
                bookmark, content, ai_result, original_folder
            )

            categorized[category].append((bookmark, subcategory))

        return dict(categorized)

    def _determine_category(
        self,
        bookmark: Bookmark,
        content: Optional[ContentData],
        ai_result: Optional[AIProcessingResult],
        original_folder: str,
    ) -> Tuple[str, str]:
        """
        Determine the category and subcategory for a bookmark.

        Returns:
            Tuple of (category, subcategory)
        """
        # Collect all text for analysis
        text_parts = []

        # Add bookmark data
        if bookmark.title:
            text_parts.append(bookmark.title.lower())
        if bookmark.note:
            text_parts.append(bookmark.note.lower())
        if bookmark.tags:
            if isinstance(bookmark.tags, list):
                tag_text = " ".join(bookmark.tags).lower()
            else:
                tag_text = str(bookmark.tags).replace('"', "").lower()
            text_parts.append(tag_text)

        # Add content data
        if content:
            if content.title:
                text_parts.append(content.title.lower())
            if content.meta_description:
                text_parts.append(content.meta_description.lower())

        # Add AI results
        if ai_result and ai_result.enhanced_description:
            text_parts.append(ai_result.enhanced_description.lower())

        # Combine all text
        full_text = " ".join(text_parts)

        # Parse domain
        domain = self._extract_domain(bookmark.url)

        # Score each category
        category_scores = {}
        subcategory_matches = {}

        for category, patterns in self.category_patterns.items():
            score = 0
            matched_subcategories = []

            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in full_text:
                    score += 2

            # Check domains
            if domain in patterns["domains"]:
                score += 5

            # Check subcategories
            for subcat, subcat_keywords in patterns.get("subcategories", {}).items():
                for keyword in subcat_keywords:
                    if keyword in full_text:
                        score += 1
                        matched_subcategories.append(subcat)

            # Bonus for original folder hint
            if original_folder and category.lower() in original_folder.lower():
                score += 3

            category_scores[category] = score
            if matched_subcategories:
                # Most frequent subcategory
                subcat_counter = Counter(matched_subcategories)
                subcategory_matches[category] = subcat_counter.most_common(1)[0][0]

        # Select best category
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:  # Has some match
                category = best_category[0]
                subcategory = subcategory_matches.get(category, "General")
            else:
                category = self._fallback_category(bookmark, original_folder)
                subcategory = "General"
        else:
            category = self._fallback_category(bookmark, original_folder)
            subcategory = "General"

        return category, subcategory

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    def _fallback_category(self, bookmark: Bookmark, original_folder: str) -> str:
        """Determine fallback category when no patterns match"""
        # Try to use original folder as hint
        if original_folder:
            # Extract first level folder
            parts = original_folder.split("/")
            if parts and parts[0]:
                # Clean up the folder name
                folder_name = parts[0].strip()
                # Check if it matches any category
                for category in self.category_patterns:
                    if (
                        folder_name.lower() in category.lower()
                        or category.lower() in folder_name.lower()
                    ):
                        return category
                # Use as-is if reasonable
                if len(folder_name) > 2 and len(folder_name) < 30:
                    return folder_name.title()

        # Ultimate fallback
        return "Uncategorized"

    def _build_folder_hierarchy(
        self,
        categorized_bookmarks: Dict[str, List[Tuple[Bookmark, str]]],
        root: FolderNode,
    ) -> FolderNode:
        """Build the folder hierarchy from categorized bookmarks"""
        # Create main category folders
        for category, bookmark_list in categorized_bookmarks.items():
            category_folder = FolderNode(name=category, path=category)
            root.add_child(category_folder)

            # Group by subcategory
            subcategory_groups = defaultdict(list)
            for bookmark, subcategory in bookmark_list:
                subcategory_groups[subcategory].append(bookmark)

            # Create subcategory folders if needed
            if (
                len(subcategory_groups) > 1
                or len(bookmark_list) > self.max_bookmarks_per_folder
            ):
                for subcategory, bookmarks in subcategory_groups.items():
                    subcat_folder = FolderNode(
                        name=subcategory, path=f"{category}/{subcategory}"
                    )
                    category_folder.add_child(subcat_folder)
                    for bookmark in bookmarks:
                        subcat_folder.add_bookmark(bookmark)
            else:
                # Add all bookmarks directly to category folder
                for bookmark, _ in bookmark_list:
                    category_folder.add_bookmark(bookmark)

        return root

    def _apply_folder_limits(self, root: FolderNode) -> None:
        """Apply folder size limits and create sub-folders as needed"""
        self._split_large_folders(root)

    def _split_large_folders(self, folder: FolderNode, depth: int = 0) -> None:
        """Recursively split folders that exceed size limit"""
        # Prevent infinite recursion
        if depth > 10:
            logging.warning(f"Maximum folder split depth reached for {folder.path}")
            return

        # First, process all children
        for child in list(
            folder.children
        ):  # Use list() to avoid modification during iteration
            self._split_large_folders(child, depth + 1)

        # Check if this folder needs splitting
        if len(folder.bookmarks) > self.max_bookmarks_per_folder:
            # Group bookmarks by similarity
            groups = self._group_similar_bookmarks(folder.bookmarks)

            # If we can't create meaningful groups (all still too large),
            # do simple numeric splitting
            if all(
                len(bookmarks) > self.max_bookmarks_per_folder
                for bookmarks in groups.values()
            ):
                groups = self._simple_numeric_split(folder.bookmarks)

            # Clear bookmarks from parent folder
            folder.bookmarks = []

            # Create sub-folders for each group
            for i, (group_name, group_bookmarks) in enumerate(groups.items()):
                if len(group_bookmarks) > 0:
                    subfolder_name = group_name if group_name else f"Group{i+1}"
                    subfolder = FolderNode(
                        name=subfolder_name, path=f"{folder.path}/{subfolder_name}"
                    )
                    folder.add_child(subfolder)

                    # Add bookmarks to subfolder
                    for bookmark in group_bookmarks:
                        subfolder.add_bookmark(bookmark)

                    # Recursively split if still too large
                    if len(group_bookmarks) > self.max_bookmarks_per_folder:
                        self._split_large_folders(subfolder, depth + 1)

    def _group_similar_bookmarks(
        self, bookmarks: List[Bookmark]
    ) -> Dict[str, List[Bookmark]]:
        """Group bookmarks by similarity for sub-folder creation"""
        groups = defaultdict(list)

        # Simple grouping by domain or common keywords
        for bookmark in bookmarks:
            # Try grouping by domain
            domain = self._extract_domain(bookmark.url)
            if domain:
                # Simplify domain for grouping
                domain_parts = domain.split(".")
                if len(domain_parts) >= 2:
                    main_domain = domain_parts[-2]  # e.g., 'github' from 'github.com'
                    groups[main_domain.title()].append(bookmark)
                else:
                    groups["Other"].append(bookmark)
            else:
                groups["Other"].append(bookmark)

        # Ensure groups aren't too small
        final_groups = {}
        other_bookmarks = []

        for group_name, group_bookmarks in groups.items():
            if len(group_bookmarks) >= 3:  # Minimum group size
                final_groups[group_name] = group_bookmarks
            else:
                other_bookmarks.extend(group_bookmarks)

        # Add remaining bookmarks to "Other" group
        if other_bookmarks:
            final_groups["Other"] = other_bookmarks

        return final_groups

    def _simple_numeric_split(
        self, bookmarks: List[Bookmark]
    ) -> Dict[str, List[Bookmark]]:
        """Split bookmarks into numbered groups when similarity grouping fails"""
        groups = {}
        group_size = self.max_bookmarks_per_folder

        for i in range(0, len(bookmarks), group_size):
            group_number = (i // group_size) + 1
            group_name = f"Part {group_number}"
            groups[group_name] = bookmarks[i : i + group_size]

        return groups

    def _assign_bookmarks_to_folders(self, root: FolderNode) -> Dict[str, str]:
        """Create the final bookmark -> folder path assignments"""
        assignments = {}
        self._collect_assignments(root, assignments)
        return assignments

    def _collect_assignments(
        self, folder: FolderNode, assignments: Dict[str, str]
    ) -> None:
        """Recursively collect bookmark assignments"""
        # Skip root folder in path
        folder_path = folder.get_full_path() if folder.name != "root" else ""

        # Assign bookmarks in this folder
        for bookmark in folder.bookmarks:
            assignments[bookmark.url] = folder_path

        # Process children
        for child in folder.children:
            self._collect_assignments(child, assignments)

    def _calculate_folder_stats(self, root: FolderNode) -> Dict[str, int]:
        """Calculate bookmark count for each folder"""
        stats = {}
        self._collect_stats(root, stats)
        return stats

    def _collect_stats(self, folder: FolderNode, stats: Dict[str, int]) -> None:
        """Recursively collect folder statistics"""
        if folder.name != "root":
            folder_path = folder.get_full_path()
            stats[folder_path] = len(folder.bookmarks)

        for child in folder.children:
            self._collect_stats(child, stats)

    def _calculate_max_depth(self, root: FolderNode, current_depth: int = 0) -> int:
        """Calculate maximum folder depth"""
        if not root.children:
            return current_depth

        max_child_depth = current_depth
        for child in root.children:
            child_depth = self._calculate_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def get_folder_report(self, result: FolderGenerationResult) -> str:
        """Generate a human-readable folder structure report"""
        lines = ["AI-Generated Folder Structure Report", "=" * 40]
        lines.append(f"Total Folders: {result.total_folders}")
        lines.append(f"Maximum Depth: {result.max_depth}")
        lines.append(f"Processing Time: {result.processing_time:.2f}s")
        lines.append("")

        # Show folder hierarchy
        lines.append("Folder Hierarchy:")
        self._add_folder_lines(result.root_folder, lines, indent="")

        return "\n".join(lines)

    def _add_folder_lines(
        self, folder: FolderNode, lines: List[str], indent: str
    ) -> None:
        """Add folder hierarchy lines to report"""
        if folder.name != "root":
            bookmark_count = len(folder.bookmarks)
            lines.append(f"{indent}{folder.name} ({bookmark_count} bookmarks)")
            indent += "  "

        for child in sorted(folder.children, key=lambda x: x.name):
            self._add_folder_lines(child, lines, indent)


@dataclass
class FolderSuggestion:
    """A folder suggestion with confidence and reasoning."""

    path: str
    confidence: float
    reasoning: str
    bookmark_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "bookmark_count": self.bookmark_count,
        }


@dataclass
class FolderSuggestionResult:
    """Results from folder suggestion mode."""

    suggestions: Dict[str, FolderSuggestion]  # url -> suggestion
    learned_patterns: Dict[str, List[str]]  # category -> folder patterns
    total_bookmarks: int
    confidence_avg: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "suggestions": {
                url: sugg.to_dict() for url, sugg in self.suggestions.items()
            },
            "learned_patterns": self.learned_patterns,
            "total_bookmarks": self.total_bookmarks,
            "confidence_avg": self.confidence_avg,
        }

    def to_json(self, file_path: str) -> None:
        """Save suggestions to JSON file."""
        import json
        from pathlib import Path

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Saved folder suggestions to: {file_path}")


class EnhancedFolderGenerator(AIFolderGenerator):
    """
    Enhanced folder organization with preservation and suggestion modes.

    Features:
    - Preserve existing folders (--preserve-folders)
    - Suggest folders without changing (--suggest-folders)
    - Learn from existing folder structure (--learn-folders)
    - Control folder depth (--max-folder-depth)
    """

    def __init__(
        self,
        max_bookmarks_per_folder: int = 20,
        ai_engine: str = "local",
        api_key: Optional[str] = None,
        preserve_existing: bool = False,
        suggest_only: bool = False,
        learn_from_existing: bool = False,
        max_depth: int = 3,
    ):
        """
        Initialize enhanced folder generator.

        Args:
            max_bookmarks_per_folder: Maximum bookmarks per folder
            ai_engine: AI engine to use
            api_key: API key for cloud services
            preserve_existing: Keep original folder assignments
            suggest_only: Only suggest folders, don't apply
            learn_from_existing: Learn patterns from existing folders
            max_depth: Maximum folder hierarchy depth
        """
        super().__init__(max_bookmarks_per_folder, ai_engine, api_key)

        self.preserve_existing = preserve_existing
        self.suggest_only = suggest_only
        self.learn_from_existing = learn_from_existing
        self.max_depth = max_depth

        # Learned patterns from existing folder structure
        self.learned_patterns: Dict[str, List[str]] = {}
        self.folder_domain_mapping: Dict[str, str] = {}

        logging.info(
            f"Enhanced folder generator initialized "
            f"(preserve={preserve_existing}, suggest={suggest_only}, "
            f"learn={learn_from_existing}, max_depth={max_depth})"
        )

    def generate_folder_structure(
        self,
        bookmarks: List[Bookmark],
        content_data_map: Optional[Dict[str, ContentData]] = None,
        ai_results_map: Optional[Dict[str, AIProcessingResult]] = None,
        original_folders_map: Optional[Dict[str, str]] = None,
    ) -> FolderGenerationResult:
        """
        Generate folder structure with enhanced options.

        Args:
            bookmarks: List of bookmarks to organize
            content_data_map: Content analysis results
            ai_results_map: AI processing results
            original_folders_map: Original folder paths as hints

        Returns:
            FolderGenerationResult with folder assignments
        """
        import time

        start_time = time.time()

        # Build original folders map from bookmarks if not provided
        if original_folders_map is None:
            original_folders_map = {}
            for bookmark in bookmarks:
                if bookmark.folder:
                    original_folders_map[bookmark.url] = bookmark.folder

        # Learn from existing structure first
        if self.learn_from_existing and original_folders_map:
            self._learn_from_existing_structure(bookmarks, original_folders_map)

        # Preserve existing folders mode
        if self.preserve_existing:
            result = self._preserve_existing_folders(
                bookmarks, original_folders_map, start_time
            )
            return result

        # Normal folder generation with learned patterns
        result = super().generate_folder_structure(
            bookmarks, content_data_map, ai_results_map, original_folders_map
        )

        # Apply max depth limit
        if self.max_depth > 0:
            result = self._apply_max_depth(result)

        return result

    def suggest_folders(
        self,
        bookmarks: List[Bookmark],
        content_data_map: Optional[Dict[str, ContentData]] = None,
        ai_results_map: Optional[Dict[str, AIProcessingResult]] = None,
        original_folders_map: Optional[Dict[str, str]] = None,
    ) -> FolderSuggestionResult:
        """
        Generate folder suggestions without applying them.

        Args:
            bookmarks: List of bookmarks to analyze
            content_data_map: Content analysis results
            ai_results_map: AI processing results
            original_folders_map: Original folder paths

        Returns:
            FolderSuggestionResult with suggestions
        """
        logging.info(f"Generating folder suggestions for {len(bookmarks)} bookmarks")

        # Build original folders map from bookmarks if not provided
        if original_folders_map is None:
            original_folders_map = {}
            for bookmark in bookmarks:
                if bookmark.folder:
                    original_folders_map[bookmark.url] = bookmark.folder

        # Learn from existing if enabled
        if self.learn_from_existing and original_folders_map:
            self._learn_from_existing_structure(bookmarks, original_folders_map)

        # Initialize data maps
        if content_data_map is None:
            content_data_map = {}
        if ai_results_map is None:
            ai_results_map = {}

        suggestions: Dict[str, FolderSuggestion] = {}
        total_confidence = 0.0

        for bookmark in bookmarks:
            content = content_data_map.get(bookmark.url)
            ai_result = ai_results_map.get(bookmark.url)
            original_folder = original_folders_map.get(bookmark.url, "")

            # Determine suggested folder
            category, subcategory = self._determine_category(
                bookmark, content, ai_result, original_folder
            )

            # Build suggested path
            if subcategory and subcategory != "General":
                suggested_path = f"{category}/{subcategory}"
            else:
                suggested_path = category

            # Limit depth
            if self.max_depth > 0:
                parts = suggested_path.split("/")
                suggested_path = "/".join(parts[: self.max_depth])

            # Calculate confidence
            confidence = self._calculate_folder_confidence(
                bookmark, content, suggested_path, original_folder
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(
                bookmark, content, category, subcategory, original_folder
            )

            suggestions[bookmark.url] = FolderSuggestion(
                path=suggested_path,
                confidence=confidence,
                reasoning=reasoning,
            )
            total_confidence += confidence

        avg_confidence = total_confidence / len(bookmarks) if bookmarks else 0.0

        return FolderSuggestionResult(
            suggestions=suggestions,
            learned_patterns=self.learned_patterns.copy(),
            total_bookmarks=len(bookmarks),
            confidence_avg=avg_confidence,
        )

    def _preserve_existing_folders(
        self,
        bookmarks: List[Bookmark],
        original_folders_map: Dict[str, str],
        start_time: float,
    ) -> FolderGenerationResult:
        """
        Preserve existing folder assignments.

        Args:
            bookmarks: List of bookmarks
            original_folders_map: Original folder paths
            start_time: Processing start time

        Returns:
            FolderGenerationResult preserving original folders
        """
        import time

        logging.info("Preserving existing folder structure")

        root = FolderNode(name="root", path="")
        folder_assignments = {}
        folder_nodes: Dict[str, FolderNode] = {}

        for bookmark in bookmarks:
            original_folder = original_folders_map.get(bookmark.url, "")

            # Limit folder depth
            if original_folder and self.max_depth > 0:
                parts = original_folder.split("/")
                original_folder = "/".join(parts[: self.max_depth])

            # Default to "Uncategorized" if no folder
            if not original_folder:
                original_folder = "Uncategorized"

            # Get or create folder node
            if original_folder not in folder_nodes:
                folder_node = self._create_folder_path(root, original_folder)
                folder_nodes[original_folder] = folder_node

            # Add bookmark to folder
            folder_nodes[original_folder].add_bookmark(bookmark)
            folder_assignments[bookmark.url] = original_folder

        # Calculate stats
        folder_stats = self._calculate_folder_stats(root)
        max_depth = self._calculate_max_depth(root)
        processing_time = time.time() - start_time

        return FolderGenerationResult(
            root_folder=root,
            folder_assignments=folder_assignments,
            total_folders=len(folder_stats),
            max_depth=max_depth,
            folder_stats=folder_stats,
            processing_time=processing_time,
        )

    def _create_folder_path(self, root: FolderNode, path: str) -> FolderNode:
        """Create folder node hierarchy for a path."""
        if not path:
            return root

        parts = path.split("/")
        current = root
        current_path = ""

        for part in parts:
            if not part:
                continue

            current_path = f"{current_path}/{part}" if current_path else part

            # Check if child exists
            existing = None
            for child in current.children:
                if child.name == part:
                    existing = child
                    break

            if existing:
                current = existing
            else:
                # Create new folder
                new_folder = FolderNode(name=part, path=current_path)
                current.add_child(new_folder)
                current = new_folder

        return current

    def _learn_from_existing_structure(
        self,
        bookmarks: List[Bookmark],
        original_folders_map: Dict[str, str],
    ) -> None:
        """
        Learn patterns from existing folder structure.

        Args:
            bookmarks: List of bookmarks
            original_folders_map: Existing folder assignments
        """
        logging.info("Learning from existing folder structure")

        # Clear existing learned patterns
        self.learned_patterns.clear()
        self.folder_domain_mapping.clear()

        # Track domain -> folder mappings
        domain_folder_count: Dict[str, Dict[str, int]] = {}

        # Track keyword -> folder mappings
        keyword_folder_count: Dict[str, Dict[str, int]] = {}

        for bookmark in bookmarks:
            folder = original_folders_map.get(bookmark.url, "")
            if not folder:
                continue

            # Extract domain
            domain = self._extract_domain(bookmark.url)
            if domain:
                if domain not in domain_folder_count:
                    domain_folder_count[domain] = {}
                folder_top = folder.split("/")[0]
                domain_folder_count[domain][folder_top] = (
                    domain_folder_count[domain].get(folder_top, 0) + 1
                )

            # Extract keywords from title and tags
            keywords = set()
            if bookmark.title:
                for word in bookmark.title.lower().split():
                    if len(word) > 3:
                        keywords.add(word)
            if bookmark.tags:
                tags = bookmark.tags if isinstance(bookmark.tags, list) else []
                for tag in tags:
                    keywords.add(tag.lower())

            folder_top = folder.split("/")[0]
            for keyword in keywords:
                if keyword not in keyword_folder_count:
                    keyword_folder_count[keyword] = {}
                keyword_folder_count[keyword][folder_top] = (
                    keyword_folder_count[keyword].get(folder_top, 0) + 1
                )

        # Build domain -> folder mapping (most common folder for each domain)
        for domain, folders in domain_folder_count.items():
            if folders:
                best_folder = max(folders.items(), key=lambda x: x[1])[0]
                # Only use if count is significant
                if folders[best_folder] >= 2:
                    self.folder_domain_mapping[domain] = best_folder

        # Build learned patterns (top folders for each keyword category)
        for keyword, folders in keyword_folder_count.items():
            if folders:
                best_folder = max(folders.items(), key=lambda x: x[1])[0]
                if folders[best_folder] >= 2:
                    if best_folder not in self.learned_patterns:
                        self.learned_patterns[best_folder] = []
                    if keyword not in self.learned_patterns[best_folder]:
                        self.learned_patterns[best_folder].append(keyword)

        logging.info(
            f"Learned {len(self.folder_domain_mapping)} domain mappings "
            f"and {len(self.learned_patterns)} folder patterns"
        )

    def _calculate_folder_confidence(
        self,
        bookmark: Bookmark,
        content: Optional[ContentData],
        suggested_path: str,
        original_folder: str,
    ) -> float:
        """
        Calculate confidence score for a folder suggestion.

        Args:
            bookmark: Bookmark being categorized
            content: Content data
            suggested_path: Suggested folder path
            original_folder: Original folder path

        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.5  # Base confidence

        # Boost if matches original folder
        if original_folder:
            original_top = original_folder.split("/")[0].lower()
            suggested_top = suggested_path.split("/")[0].lower()
            if original_top == suggested_top:
                confidence += 0.25
            elif original_top in suggested_top or suggested_top in original_top:
                confidence += 0.15

        # Boost if domain is learned
        domain = self._extract_domain(bookmark.url)
        if domain in self.folder_domain_mapping:
            if self.folder_domain_mapping[domain].lower() in suggested_path.lower():
                confidence += 0.2

        # Boost if content categories match
        if content and content.content_categories:
            for cat in content.content_categories:
                if cat.lower() in suggested_path.lower():
                    confidence += 0.1
                    break

        return min(1.0, confidence)

    def _generate_reasoning(
        self,
        bookmark: Bookmark,
        content: Optional[ContentData],
        category: str,
        subcategory: str,
        original_folder: str,
    ) -> str:
        """
        Generate human-readable reasoning for folder suggestion.

        Args:
            bookmark: Bookmark being categorized
            content: Content data
            category: Suggested category
            subcategory: Suggested subcategory
            original_folder: Original folder path

        Returns:
            Reasoning string
        """
        reasons = []

        # Domain-based reasoning
        domain = self._extract_domain(bookmark.url)
        if domain in self.folder_domain_mapping:
            reasons.append(f"Domain '{domain}' typically mapped to this folder")

        # Category pattern reasoning
        for pattern_folder, keywords in self.learned_patterns.items():
            if pattern_folder.lower() == category.lower():
                matching_keywords = []
                title_words = bookmark.title.lower().split() if bookmark.title else []
                for kw in keywords[:3]:  # Limit to first 3
                    if kw in title_words:
                        matching_keywords.append(kw)
                if matching_keywords:
                    reasons.append(f"Title contains keywords: {', '.join(matching_keywords)}")
                break

        # Content category reasoning
        if content and content.content_categories:
            matching_cats = [c for c in content.content_categories if c.lower() in category.lower()]
            if matching_cats:
                reasons.append(f"Content categorized as: {', '.join(matching_cats[:2])}")

        # Original folder reasoning
        if original_folder:
            original_top = original_folder.split("/")[0]
            if original_top.lower() in category.lower():
                reasons.append(f"Similar to original folder '{original_top}'")

        if not reasons:
            reasons.append("Best match based on content analysis")

        return "; ".join(reasons)

    def _apply_max_depth(self, result: FolderGenerationResult) -> FolderGenerationResult:
        """
        Apply maximum depth limit to folder result.

        Args:
            result: Original folder generation result

        Returns:
            Result with depth-limited paths
        """
        if self.max_depth <= 0:
            return result

        # Limit depth in assignments
        new_assignments = {}
        for url, path in result.folder_assignments.items():
            parts = path.split("/")
            new_path = "/".join(parts[: self.max_depth])
            new_assignments[url] = new_path

        # Rebuild folder stats
        new_stats = {}
        for path, count in result.folder_stats.items():
            parts = path.split("/")
            new_path = "/".join(parts[: self.max_depth])
            new_stats[new_path] = new_stats.get(new_path, 0) + count

        return FolderGenerationResult(
            root_folder=result.root_folder,
            folder_assignments=new_assignments,
            total_folders=len(new_stats),
            max_depth=min(result.max_depth, self.max_depth),
            folder_stats=new_stats,
            processing_time=result.processing_time,
        )

    def _determine_category(
        self,
        bookmark: Bookmark,
        content: Optional[ContentData],
        ai_result: Optional[AIProcessingResult],
        original_folder: str,
    ) -> Tuple[str, str]:
        """
        Determine category using learned patterns first, then fallback.

        Overrides parent method to incorporate learned patterns.
        """
        # Check domain mapping first
        domain = self._extract_domain(bookmark.url)
        if domain in self.folder_domain_mapping:
            folder = self.folder_domain_mapping[domain]
            return folder, "General"

        # Check learned patterns
        text_parts = []
        if bookmark.title:
            text_parts.append(bookmark.title.lower())
        if bookmark.tags:
            if isinstance(bookmark.tags, list):
                text_parts.extend([t.lower() for t in bookmark.tags])

        full_text = " ".join(text_parts)

        for folder, keywords in self.learned_patterns.items():
            matching = sum(1 for kw in keywords if kw in full_text)
            if matching >= 2:  # At least 2 keyword matches
                return folder, "General"

        # Fallback to parent implementation
        return super()._determine_category(bookmark, content, ai_result, original_folder)
