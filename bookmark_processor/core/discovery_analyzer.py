"""
Discovery Analyzer for Bookmark Collections.

Analyzes bookmark data to discover patterns, suggest new tags and folders
that aren't in the built-in vocabulary. Provides insights into the collection
without modifying any data.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .data_models import Bookmark
from .tag_generator import CorpusAwareTagGenerator
from .folder_generator import AIFolderGenerator


@dataclass
class DiscoveredTerm:
    """A term discovered from bookmark analysis."""

    term: str
    frequency: int
    sources: Set[str] = field(default_factory=set)  # 'title', 'note', 'url', 'tag'
    example_bookmarks: List[str] = field(default_factory=list)  # URLs for context
    in_builtin_vocabulary: bool = False

    @property
    def source_list(self) -> str:
        """Get sources as comma-separated string."""
        return ", ".join(sorted(self.sources))


@dataclass
class DomainStats:
    """Statistics for a domain."""

    domain: str
    count: int
    current_folders: Set[str] = field(default_factory=set)
    example_titles: List[str] = field(default_factory=list)


@dataclass
class DiscoveryReport:
    """Complete discovery analysis report."""

    # Summary stats
    total_bookmarks: int = 0
    total_unique_terms: int = 0
    total_domains: int = 0

    # Discovered terms not in vocabulary
    new_tag_suggestions: List[DiscoveredTerm] = field(default_factory=list)

    # Domain analysis
    top_domains: List[DomainStats] = field(default_factory=list)
    domains_without_category: List[DomainStats] = field(default_factory=list)

    # Folder analysis
    current_folders: Dict[str, int] = field(default_factory=dict)
    suggested_new_folders: List[Dict[str, Any]] = field(default_factory=list)

    # Tag analysis
    existing_tag_frequency: Dict[str, int] = field(default_factory=dict)
    underused_builtin_tags: List[str] = field(default_factory=list)

    # Quality indicators
    bookmarks_without_tags: int = 0
    bookmarks_without_folders: int = 0
    bookmarks_with_short_titles: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "summary": {
                "total_bookmarks": self.total_bookmarks,
                "total_unique_terms": self.total_unique_terms,
                "total_domains": self.total_domains,
                "bookmarks_without_tags": self.bookmarks_without_tags,
                "bookmarks_without_folders": self.bookmarks_without_folders,
            },
            "new_tag_suggestions": [
                {
                    "term": t.term,
                    "frequency": t.frequency,
                    "sources": list(t.sources),
                    "examples": t.example_bookmarks[:3],
                }
                for t in self.new_tag_suggestions
            ],
            "top_domains": [
                {
                    "domain": d.domain,
                    "count": d.count,
                    "folders": list(d.current_folders),
                }
                for d in self.top_domains
            ],
            "domains_needing_category": [
                {
                    "domain": d.domain,
                    "count": d.count,
                    "example_titles": d.example_titles[:3],
                }
                for d in self.domains_without_category
            ],
            "current_folders": self.current_folders,
            "suggested_new_folders": self.suggested_new_folders,
            "existing_tag_frequency": dict(
                sorted(
                    self.existing_tag_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:50]
            ),
            "underused_builtin_tags": self.underused_builtin_tags,
        }


class DiscoveryAnalyzer:
    """
    Analyzes bookmark collections to discover patterns and suggest improvements.

    This analyzer reads bookmark data without modifying it and identifies:
    - High-frequency terms not in the built-in tag vocabulary
    - Domains that could benefit from dedicated folders
    - Patterns that suggest new category groupings
    - Quality issues (missing tags, folders, etc.)
    """

    # Minimum frequency for a term to be considered significant
    MIN_TERM_FREQUENCY = 3

    # Minimum frequency for a domain to be reported
    MIN_DOMAIN_FREQUENCY = 2

    # Maximum number of suggestions to return
    MAX_SUGGESTIONS = 30

    def __init__(
        self,
        min_term_frequency: int = MIN_TERM_FREQUENCY,
        min_domain_frequency: int = MIN_DOMAIN_FREQUENCY,
        max_suggestions: int = MAX_SUGGESTIONS,
    ):
        """
        Initialize the discovery analyzer.

        Args:
            min_term_frequency: Minimum occurrences for term to be suggested
            min_domain_frequency: Minimum occurrences for domain analysis
            max_suggestions: Maximum number of suggestions per category
        """
        self.min_term_frequency = min_term_frequency
        self.min_domain_frequency = min_domain_frequency
        self.max_suggestions = max_suggestions

        # Get built-in vocabularies from existing generators
        self._init_builtin_vocabulary()

    def _init_builtin_vocabulary(self) -> None:
        """Initialize the set of known vocabulary terms."""
        self.builtin_tags: Set[str] = set()
        self.builtin_domains: Set[str] = set()
        self.builtin_categories: Set[str] = set()

        # Get tag vocabulary from CorpusAwareTagGenerator
        self.builtin_tags.update(CorpusAwareTagGenerator.STOP_WORDS)

        for keywords in CorpusAwareTagGenerator.TECH_KEYWORDS.values():
            self.builtin_tags.update(k.lower() for k in keywords)

        for indicators in CorpusAwareTagGenerator.CONTENT_TYPE_INDICATORS.values():
            self.builtin_tags.update(i.lower() for i in indicators)

        # Add content type names themselves
        self.builtin_tags.update(
            k.lower() for k in CorpusAwareTagGenerator.CONTENT_TYPE_INDICATORS.keys()
        )
        self.builtin_tags.update(
            k.lower() for k in CorpusAwareTagGenerator.TECH_KEYWORDS.keys()
        )

        # Get folder vocabulary from AIFolderGenerator
        folder_gen = AIFolderGenerator()
        for category, data in folder_gen.category_patterns.items():
            self.builtin_categories.add(category.lower())
            self.builtin_tags.update(k.lower() for k in data.get("keywords", []))
            self.builtin_domains.update(d.lower() for d in data.get("domains", []))

            for subcat, keywords in data.get("subcategories", {}).items():
                self.builtin_categories.add(subcat.lower())
                self.builtin_tags.update(k.lower() for k in keywords)

    def analyze(self, bookmarks: List[Bookmark]) -> DiscoveryReport:
        """
        Analyze a collection of bookmarks for patterns and suggestions.

        Args:
            bookmarks: List of Bookmark objects to analyze

        Returns:
            DiscoveryReport with analysis results
        """
        report = DiscoveryReport(total_bookmarks=len(bookmarks))

        if not bookmarks:
            return report

        # Collect data
        term_counter: Counter = Counter()
        term_sources: Dict[str, Set[str]] = defaultdict(set)
        term_examples: Dict[str, List[str]] = defaultdict(list)
        domain_counter: Counter = Counter()
        domain_folders: Dict[str, Set[str]] = defaultdict(set)
        domain_titles: Dict[str, List[str]] = defaultdict(list)
        folder_counter: Counter = Counter()
        tag_counter: Counter = Counter()

        for bookmark in bookmarks:
            # Quality checks
            if not bookmark.tags:
                report.bookmarks_without_tags += 1
            if not bookmark.folder or bookmark.folder.lower() in ("", "unsorted"):
                report.bookmarks_without_folders += 1
            if len(bookmark.title) < 10:
                report.bookmarks_with_short_titles += 1

            # Extract terms from title
            title_terms = self._extract_terms(bookmark.title)
            for term in title_terms:
                term_counter[term] += 1
                term_sources[term].add("title")
                if len(term_examples[term]) < 5:
                    term_examples[term].append(bookmark.url)

            # Extract terms from note
            note_terms = self._extract_terms(bookmark.note)
            for term in note_terms:
                term_counter[term] += 1
                term_sources[term].add("note")
                if len(term_examples[term]) < 5:
                    term_examples[term].append(bookmark.url)

            # Extract terms from URL path
            url_terms = self._extract_url_terms(bookmark.url)
            for term in url_terms:
                term_counter[term] += 1
                term_sources[term].add("url")
                if len(term_examples[term]) < 5:
                    term_examples[term].append(bookmark.url)

            # Count existing tags
            for tag in bookmark.tags:
                tag_lower = tag.lower().strip()
                if tag_lower:
                    tag_counter[tag_lower] += 1
                    term_counter[tag_lower] += 1
                    term_sources[tag_lower].add("tag")

            # Analyze domain
            domain = self._extract_domain(bookmark.url)
            if domain:
                domain_counter[domain] += 1
                if bookmark.folder:
                    domain_folders[domain].add(bookmark.folder)
                if len(domain_titles[domain]) < 5:
                    domain_titles[domain].append(bookmark.title)

            # Count folders
            if bookmark.folder:
                folder_counter[bookmark.folder] += 1

        # Build report
        report.total_unique_terms = len(term_counter)
        report.total_domains = len(domain_counter)
        report.existing_tag_frequency = dict(tag_counter)
        report.current_folders = dict(folder_counter)

        # Find new tag suggestions (terms not in vocabulary)
        new_terms = []
        for term, freq in term_counter.most_common():
            if freq < self.min_term_frequency:
                break
            if self._is_new_term(term):
                new_terms.append(
                    DiscoveredTerm(
                        term=term,
                        frequency=freq,
                        sources=term_sources[term],
                        example_bookmarks=term_examples[term][:3],
                        in_builtin_vocabulary=False,
                    )
                )
                if len(new_terms) >= self.max_suggestions:
                    break

        report.new_tag_suggestions = new_terms

        # Analyze domains
        for domain, count in domain_counter.most_common(self.max_suggestions):
            if count < self.min_domain_frequency:
                break
            stats = DomainStats(
                domain=domain,
                count=count,
                current_folders=domain_folders[domain],
                example_titles=domain_titles[domain][:3],
            )
            report.top_domains.append(stats)

            # Check if domain has a built-in category
            if domain.lower() not in self.builtin_domains:
                report.domains_without_category.append(stats)

        # Suggest new folders based on high-frequency domains without categories
        report.suggested_new_folders = self._suggest_folders(
            report.domains_without_category, term_counter
        )

        # Find underused built-in tags
        used_builtin = set(tag_counter.keys()) & self.builtin_tags
        unused_builtin = self.builtin_tags - set(tag_counter.keys()) - CorpusAwareTagGenerator.STOP_WORDS
        # Only suggest tags that appear in the content but aren't used as tags
        potential_tags = []
        for tag in unused_builtin:
            if tag in term_counter and term_counter[tag] >= self.min_term_frequency:
                potential_tags.append((tag, term_counter[tag]))
        potential_tags.sort(key=lambda x: x[1], reverse=True)
        report.underused_builtin_tags = [t[0] for t in potential_tags[:20]]

        return report

    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text."""
        if not text:
            return []

        # Normalize and split
        text = text.lower()
        # Split on non-alphanumeric, keeping hyphens within words
        words = re.findall(r"[a-z][a-z0-9\-]*[a-z0-9]|[a-z]", text)

        # Filter
        terms = []
        for word in words:
            # Skip very short or very long terms
            if len(word) < 2 or len(word) > 30:
                continue
            # Skip pure numbers
            if word.isdigit():
                continue
            # Skip stop words
            if word in CorpusAwareTagGenerator.STOP_WORDS:
                continue
            terms.append(word)

        return terms

    def _extract_url_terms(self, url: str) -> List[str]:
        """Extract meaningful terms from URL path."""
        if not url:
            return []

        try:
            parsed = urlparse(url)
            path = parsed.path

            # Split path and extract terms
            parts = re.split(r"[/\-_.]", path)
            terms = []
            for part in parts:
                part = part.lower().strip()
                if len(part) >= 3 and not part.isdigit():
                    if part not in CorpusAwareTagGenerator.STOP_WORDS:
                        terms.append(part)
            return terms
        except Exception:
            return []

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        if not url:
            return None

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain if domain else None
        except Exception:
            return None

    def _is_new_term(self, term: str) -> bool:
        """Check if a term is not in the built-in vocabulary."""
        term_lower = term.lower()

        # Skip if it's a stop word
        if term_lower in CorpusAwareTagGenerator.STOP_WORDS:
            return False

        # Skip if it's already in built-in tags
        if term_lower in self.builtin_tags:
            return False

        # Skip very common web terms
        common_web_terms = {
            "http", "https", "www", "com", "org", "net", "html", "php",
            "index", "page", "post", "blog", "article", "view", "read",
            "click", "link", "more", "see", "also", "new", "best",
        }
        if term_lower in common_web_terms:
            return False

        return True

    def _suggest_folders(
        self,
        uncategorized_domains: List[DomainStats],
        term_frequency: Counter,
    ) -> List[Dict[str, Any]]:
        """Suggest new folders based on domain and term patterns."""
        suggestions = []

        # Group domains by potential category
        domain_groups: Dict[str, List[DomainStats]] = defaultdict(list)

        for domain_stat in uncategorized_domains:
            # Try to identify a grouping
            domain = domain_stat.domain

            # Check for common patterns
            if any(x in domain for x in ["github", "gitlab", "bitbucket"]):
                domain_groups["Code Repositories"].append(domain_stat)
            elif any(x in domain for x in ["medium", "substack", "blog"]):
                domain_groups["Blogs & Articles"].append(domain_stat)
            elif any(x in domain for x in ["youtube", "vimeo", "video"]):
                domain_groups["Video Content"].append(domain_stat)
            elif any(x in domain for x in ["docs", "documentation", "wiki"]):
                domain_groups["Documentation"].append(domain_stat)
            elif domain_stat.count >= 5:
                # High-frequency domain could be its own folder
                domain_groups[f"From {domain}"].append(domain_stat)

        # Convert to suggestions
        for folder_name, domains in domain_groups.items():
            if len(domains) >= 1:
                total_bookmarks = sum(d.count for d in domains)
                suggestions.append({
                    "suggested_folder": folder_name,
                    "domains": [d.domain for d in domains],
                    "bookmark_count": total_bookmarks,
                    "rationale": f"Groups {len(domains)} domain(s) with {total_bookmarks} bookmarks",
                })

        # Sort by bookmark count
        suggestions.sort(key=lambda x: x["bookmark_count"], reverse=True)
        return suggestions[:self.max_suggestions]


def format_discovery_report(report: DiscoveryReport, verbose: bool = False) -> str:
    """
    Format a discovery report as human-readable text.

    Args:
        report: The DiscoveryReport to format
        verbose: Include additional details

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BOOKMARK COLLECTION DISCOVERY REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total bookmarks analyzed: {report.total_bookmarks}")
    lines.append(f"Unique terms found: {report.total_unique_terms}")
    lines.append(f"Unique domains: {report.total_domains}")
    lines.append(f"Current folders: {len(report.current_folders)}")
    lines.append("")

    # Quality indicators
    lines.append("QUALITY INDICATORS")
    lines.append("-" * 40)
    pct_no_tags = (
        report.bookmarks_without_tags / report.total_bookmarks * 100
        if report.total_bookmarks > 0
        else 0
    )
    pct_no_folders = (
        report.bookmarks_without_folders / report.total_bookmarks * 100
        if report.total_bookmarks > 0
        else 0
    )
    lines.append(f"Bookmarks without tags: {report.bookmarks_without_tags} ({pct_no_tags:.1f}%)")
    lines.append(f"Bookmarks without folders: {report.bookmarks_without_folders} ({pct_no_folders:.1f}%)")
    lines.append(f"Bookmarks with short titles: {report.bookmarks_with_short_titles}")
    lines.append("")

    # New tag suggestions
    if report.new_tag_suggestions:
        lines.append("SUGGESTED NEW TAGS")
        lines.append("-" * 40)
        lines.append("These terms appear frequently but aren't in the built-in vocabulary:")
        lines.append("")
        for term in report.new_tag_suggestions[:15]:
            lines.append(f"  {term.term:<25} (found {term.frequency}x in {term.source_list})")
        if len(report.new_tag_suggestions) > 15:
            lines.append(f"  ... and {len(report.new_tag_suggestions) - 15} more")
        lines.append("")

    # Underused built-in tags
    if report.underused_builtin_tags:
        lines.append("UNDERUSED BUILT-IN TAGS")
        lines.append("-" * 40)
        lines.append("These terms appear in your content but aren't used as tags:")
        lines.append("")
        lines.append(f"  {', '.join(report.underused_builtin_tags[:15])}")
        lines.append("")

    # Top domains
    if report.top_domains:
        lines.append("TOP DOMAINS")
        lines.append("-" * 40)
        for domain_stat in report.top_domains[:10]:
            folders = ", ".join(domain_stat.current_folders) if domain_stat.current_folders else "(no folder)"
            lines.append(f"  {domain_stat.domain:<30} {domain_stat.count:>4} bookmarks  [{folders}]")
        lines.append("")

    # Domains needing categories
    if report.domains_without_category:
        lines.append("DOMAINS WITHOUT BUILT-IN CATEGORY")
        lines.append("-" * 40)
        lines.append("These frequent domains aren't in the built-in category list:")
        lines.append("")
        for domain_stat in report.domains_without_category[:10]:
            lines.append(f"  {domain_stat.domain:<30} ({domain_stat.count} bookmarks)")
        lines.append("")

    # Suggested new folders
    if report.suggested_new_folders:
        lines.append("SUGGESTED NEW FOLDERS")
        lines.append("-" * 40)
        for suggestion in report.suggested_new_folders[:10]:
            lines.append(f"  {suggestion['suggested_folder']}")
            lines.append(f"    Domains: {', '.join(suggestion['domains'][:5])}")
            lines.append(f"    Would contain: {suggestion['bookmark_count']} bookmarks")
            lines.append("")

    # Current folder distribution
    if verbose and report.current_folders:
        lines.append("CURRENT FOLDER DISTRIBUTION")
        lines.append("-" * 40)
        sorted_folders = sorted(
            report.current_folders.items(), key=lambda x: x[1], reverse=True
        )
        for folder, count in sorted_folders[:20]:
            lines.append(f"  {folder:<40} {count:>4}")
        lines.append("")

    # Existing tag frequency
    if verbose and report.existing_tag_frequency:
        lines.append("TOP EXISTING TAGS")
        lines.append("-" * 40)
        sorted_tags = sorted(
            report.existing_tag_frequency.items(), key=lambda x: x[1], reverse=True
        )
        for tag, count in sorted_tags[:20]:
            lines.append(f"  {tag:<30} {count:>4}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)
