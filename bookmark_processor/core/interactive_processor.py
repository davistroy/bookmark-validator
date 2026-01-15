"""
Interactive Bookmark Processing Module

Provides interactive approval mode for bookmark processing, allowing users
to review and approve proposed changes before they are applied.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.text import Text
    from rich.style import Style

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .data_models import Bookmark
from .ai_processor import AIProcessingResult


class InteractiveAction(str, Enum):
    """Actions available during interactive processing."""

    ACCEPT_ALL = "a"
    DESCRIPTION_ONLY = "d"
    TAGS_ONLY = "t"
    FOLDER_ONLY = "f"
    SKIP = "s"
    QUIT = "q"
    EDIT_DESCRIPTION = "e"
    EDIT_TAGS = "T"
    EDIT_FOLDER = "F"
    UNDO = "u"
    HELP = "h"


@dataclass
class ProposedChanges:
    """Container for proposed changes to a bookmark."""

    url: str
    original_description: str
    proposed_description: str
    description_confidence: float
    description_method: str

    original_tags: List[str]
    proposed_tags: List[str]
    tags_confidence: float

    original_folder: str
    proposed_folder: str
    folder_confidence: float

    overall_confidence: float = 0.0

    def __post_init__(self):
        """Calculate overall confidence after initialization."""
        if self.overall_confidence == 0.0:
            # Weighted average of confidences
            self.overall_confidence = (
                self.description_confidence * 0.4
                + self.tags_confidence * 0.3
                + self.folder_confidence * 0.3
            )

    def has_description_change(self) -> bool:
        """Check if description changed."""
        return self.proposed_description != self.original_description

    def has_tags_change(self) -> bool:
        """Check if tags changed."""
        return set(self.proposed_tags) != set(self.original_tags)

    def has_folder_change(self) -> bool:
        """Check if folder changed."""
        return self.proposed_folder != self.original_folder

    def has_any_change(self) -> bool:
        """Check if any change is proposed."""
        return (
            self.has_description_change()
            or self.has_tags_change()
            or self.has_folder_change()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "original_description": self.original_description,
            "proposed_description": self.proposed_description,
            "description_confidence": self.description_confidence,
            "description_method": self.description_method,
            "original_tags": self.original_tags,
            "proposed_tags": self.proposed_tags,
            "tags_confidence": self.tags_confidence,
            "original_folder": self.original_folder,
            "proposed_folder": self.proposed_folder,
            "folder_confidence": self.folder_confidence,
            "overall_confidence": self.overall_confidence,
        }


@dataclass
class ProcessedBookmark:
    """Result of processing a single bookmark interactively."""

    bookmark: Bookmark
    changes_applied: List[str]  # 'description', 'tags', 'folder'
    action_taken: InteractiveAction
    original_state: Dict[str, Any]
    was_modified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.bookmark.url,
            "changes_applied": self.changes_applied,
            "action_taken": self.action_taken.value,
            "was_modified": self.was_modified,
        }


@dataclass
class InteractiveSessionStats:
    """Statistics for an interactive processing session."""

    total_bookmarks: int = 0
    processed_count: int = 0
    accepted_all: int = 0
    description_only: int = 0
    tags_only: int = 0
    folder_only: int = 0
    skipped: int = 0
    edited: int = 0
    auto_accepted: int = 0  # Above threshold, auto-approved

    def get_progress_percentage(self) -> float:
        """Get processing progress as percentage."""
        if self.total_bookmarks == 0:
            return 0.0
        return (self.processed_count / self.total_bookmarks) * 100

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "total_bookmarks": self.total_bookmarks,
            "processed_count": self.processed_count,
            "accepted_all": self.accepted_all,
            "description_only": self.description_only,
            "tags_only": self.tags_only,
            "folder_only": self.folder_only,
            "skipped": self.skipped,
            "edited": self.edited,
            "auto_accepted": self.auto_accepted,
        }


class InteractiveProcessor:
    """
    Process bookmarks with interactive user approval.

    Allows users to review proposed changes (description, tags, folder)
    and selectively approve or modify them.
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        confirm_threshold: float = 0.0,
        show_diff: bool = True,
        compact_mode: bool = False,
        auto_save_interval: int = 10,
        console: Optional["Console"] = None,
    ):
        """
        Initialize interactive processor.

        Args:
            pipeline: BookmarkProcessingPipeline instance (or None for standalone use)
            confirm_threshold: Confidence threshold above which to auto-accept (0 = confirm all)
            show_diff: Whether to show before/after comparison
            compact_mode: Use compact display mode
            auto_save_interval: Save progress every N bookmarks
            console: Rich console instance (creates one if not provided)
        """
        self.pipeline = pipeline
        self.confirm_threshold = confirm_threshold
        self.show_diff = show_diff
        self.compact_mode = compact_mode
        self.auto_save_interval = auto_save_interval

        # Initialize console
        if RICH_AVAILABLE:
            self.console = console or Console()
        else:
            self.console = None

        # Session state
        self.stats = InteractiveSessionStats()
        self.history: List[ProcessedBookmark] = []
        self.pending_changes: Dict[str, ProposedChanges] = {}

        # Callbacks
        self._on_progress: Optional[Callable[[InteractiveSessionStats], None]] = None
        self._on_save: Optional[Callable[[List[ProcessedBookmark]], None]] = None

        logging.info(
            f"InteractiveProcessor initialized (threshold={confirm_threshold}, "
            f"show_diff={show_diff}, compact={compact_mode})"
        )

    def set_on_progress(
        self, callback: Callable[[InteractiveSessionStats], None]
    ) -> None:
        """Set callback for progress updates."""
        self._on_progress = callback

    def set_on_save(
        self, callback: Callable[[List[ProcessedBookmark]], None]
    ) -> None:
        """Set callback for save events."""
        self._on_save = callback

    def process_interactive(
        self,
        bookmarks: List[Bookmark],
        proposed_changes: Optional[Dict[str, ProposedChanges]] = None,
    ) -> List[ProcessedBookmark]:
        """
        Process bookmarks with interactive approval.

        Args:
            bookmarks: List of bookmarks to process
            proposed_changes: Optional pre-computed changes (if not using pipeline)

        Returns:
            List of ProcessedBookmark results
        """
        if not bookmarks:
            logging.info("No bookmarks to process")
            return []

        self.stats = InteractiveSessionStats(total_bookmarks=len(bookmarks))
        self.history = []
        self.pending_changes = proposed_changes or {}

        results: List[ProcessedBookmark] = []

        self._display_welcome(len(bookmarks))

        for i, bookmark in enumerate(bookmarks):
            # Get or compute proposed changes
            changes = self._get_or_compute_changes(bookmark)

            if changes is None:
                # No changes proposed, skip
                result = ProcessedBookmark(
                    bookmark=bookmark,
                    changes_applied=[],
                    action_taken=InteractiveAction.SKIP,
                    original_state=self._capture_state(bookmark),
                    was_modified=False,
                )
                results.append(result)
                self.stats.processed_count += 1
                self.stats.skipped += 1
                continue

            # Check if auto-accept based on confidence threshold
            if (
                self.confirm_threshold > 0
                and changes.overall_confidence >= self.confirm_threshold
            ):
                result = self._auto_accept(bookmark, changes)
                results.append(result)
                self.history.append(result)
                self.stats.auto_accepted += 1
                self.stats.processed_count += 1
                continue

            # Display bookmark and get user decision
            self._display_bookmark(i, len(bookmarks), bookmark, changes)

            action = self._prompt_action()

            if action == InteractiveAction.QUIT:
                self._display_quit_message(results)
                break

            if action == InteractiveAction.HELP:
                self._display_help()
                # Re-prompt for this bookmark
                action = self._prompt_action()

            if action == InteractiveAction.UNDO and self.history:
                self._undo_last()
                # Re-process this bookmark
                continue

            result = self._apply_action(bookmark, changes, action)
            results.append(result)
            self.history.append(result)
            self.stats.processed_count += 1

            # Update stats based on action
            self._update_stats(action)

            # Progress callback
            if self._on_progress:
                self._on_progress(self.stats)

            # Auto-save
            if (
                self.auto_save_interval > 0
                and self.stats.processed_count % self.auto_save_interval == 0
            ):
                self._trigger_save(results)

        self._display_summary()

        return results

    def propose_changes(
        self,
        bookmark: Bookmark,
        ai_result: Optional[AIProcessingResult] = None,
        proposed_tags: Optional[List[str]] = None,
        proposed_folder: Optional[str] = None,
    ) -> ProposedChanges:
        """
        Create proposed changes for a bookmark.

        Args:
            bookmark: Bookmark to propose changes for
            ai_result: Optional AI processing result
            proposed_tags: Optional proposed tags
            proposed_folder: Optional proposed folder

        Returns:
            ProposedChanges instance
        """
        # Original values
        original_description = bookmark.get_effective_description()
        original_tags = bookmark.tags or []
        original_folder = bookmark.folder or ""

        # Proposed values
        if ai_result:
            proposed_description = ai_result.enhanced_description
            description_confidence = ai_result.confidence_score
            description_method = ai_result.processing_method
        else:
            proposed_description = original_description
            description_confidence = 1.0
            description_method = "unchanged"

        tags = proposed_tags if proposed_tags is not None else original_tags
        tags_confidence = 0.8 if proposed_tags is not None else 1.0

        folder = proposed_folder if proposed_folder is not None else original_folder
        folder_confidence = 0.7 if proposed_folder is not None else 1.0

        return ProposedChanges(
            url=bookmark.url,
            original_description=original_description,
            proposed_description=proposed_description,
            description_confidence=description_confidence,
            description_method=description_method,
            original_tags=original_tags,
            proposed_tags=tags,
            tags_confidence=tags_confidence,
            original_folder=original_folder,
            proposed_folder=folder,
            folder_confidence=folder_confidence,
        )

    def _get_or_compute_changes(
        self, bookmark: Bookmark
    ) -> Optional[ProposedChanges]:
        """Get cached changes or compute new ones."""
        if bookmark.url in self.pending_changes:
            return self.pending_changes[bookmark.url]

        if self.pipeline:
            # Use pipeline to compute changes
            changes = self._compute_changes_via_pipeline(bookmark)
            if changes:
                self.pending_changes[bookmark.url] = changes
            return changes

        # No changes available
        return None

    def _compute_changes_via_pipeline(
        self, bookmark: Bookmark
    ) -> Optional[ProposedChanges]:
        """Compute changes using the pipeline."""
        try:
            # This would integrate with the actual pipeline
            # For now, return None to indicate no changes
            return None
        except Exception as e:
            logging.error(f"Error computing changes for {bookmark.url}: {e}")
            return None

    def _capture_state(self, bookmark: Bookmark) -> Dict[str, Any]:
        """Capture current bookmark state for undo."""
        return {
            "note": bookmark.note,
            "enhanced_description": bookmark.enhanced_description,
            "tags": bookmark.tags.copy() if bookmark.tags else [],
            "optimized_tags": (
                bookmark.optimized_tags.copy() if bookmark.optimized_tags else []
            ),
            "folder": bookmark.folder,
        }

    def _restore_state(self, bookmark: Bookmark, state: Dict[str, Any]) -> None:
        """Restore bookmark to previous state."""
        bookmark.note = state["note"]
        bookmark.enhanced_description = state["enhanced_description"]
        bookmark.tags = state["tags"]
        bookmark.optimized_tags = state["optimized_tags"]
        bookmark.folder = state["folder"]

    def _auto_accept(
        self, bookmark: Bookmark, changes: ProposedChanges
    ) -> ProcessedBookmark:
        """Auto-accept changes above confidence threshold."""
        original_state = self._capture_state(bookmark)
        changes_applied = []

        if changes.has_description_change():
            bookmark.enhanced_description = changes.proposed_description
            changes_applied.append("description")

        if changes.has_tags_change():
            bookmark.optimized_tags = changes.proposed_tags
            changes_applied.append("tags")

        if changes.has_folder_change():
            bookmark.folder = changes.proposed_folder
            changes_applied.append("folder")

        return ProcessedBookmark(
            bookmark=bookmark,
            changes_applied=changes_applied,
            action_taken=InteractiveAction.ACCEPT_ALL,
            original_state=original_state,
            was_modified=len(changes_applied) > 0,
        )

    def _apply_action(
        self,
        bookmark: Bookmark,
        changes: ProposedChanges,
        action: InteractiveAction,
    ) -> ProcessedBookmark:
        """Apply user action to bookmark."""
        original_state = self._capture_state(bookmark)
        changes_applied = []

        if action == InteractiveAction.ACCEPT_ALL:
            if changes.has_description_change():
                bookmark.enhanced_description = changes.proposed_description
                changes_applied.append("description")
            if changes.has_tags_change():
                bookmark.optimized_tags = changes.proposed_tags
                changes_applied.append("tags")
            if changes.has_folder_change():
                bookmark.folder = changes.proposed_folder
                changes_applied.append("folder")

        elif action == InteractiveAction.DESCRIPTION_ONLY:
            if changes.has_description_change():
                bookmark.enhanced_description = changes.proposed_description
                changes_applied.append("description")

        elif action == InteractiveAction.TAGS_ONLY:
            if changes.has_tags_change():
                bookmark.optimized_tags = changes.proposed_tags
                changes_applied.append("tags")

        elif action == InteractiveAction.FOLDER_ONLY:
            if changes.has_folder_change():
                bookmark.folder = changes.proposed_folder
                changes_applied.append("folder")

        elif action == InteractiveAction.EDIT_DESCRIPTION:
            new_description = self._prompt_edit_description(
                changes.proposed_description
            )
            if new_description:
                bookmark.enhanced_description = new_description
                changes_applied.append("description")

        elif action == InteractiveAction.EDIT_TAGS:
            new_tags = self._prompt_edit_tags(changes.proposed_tags)
            if new_tags is not None:
                bookmark.optimized_tags = new_tags
                changes_applied.append("tags")

        elif action == InteractiveAction.EDIT_FOLDER:
            new_folder = self._prompt_edit_folder(changes.proposed_folder)
            if new_folder is not None:
                bookmark.folder = new_folder
                changes_applied.append("folder")

        return ProcessedBookmark(
            bookmark=bookmark,
            changes_applied=changes_applied,
            action_taken=action,
            original_state=original_state,
            was_modified=len(changes_applied) > 0,
        )

    def _update_stats(self, action: InteractiveAction) -> None:
        """Update session statistics based on action."""
        if action == InteractiveAction.ACCEPT_ALL:
            self.stats.accepted_all += 1
        elif action == InteractiveAction.DESCRIPTION_ONLY:
            self.stats.description_only += 1
        elif action == InteractiveAction.TAGS_ONLY:
            self.stats.tags_only += 1
        elif action == InteractiveAction.FOLDER_ONLY:
            self.stats.folder_only += 1
        elif action == InteractiveAction.SKIP:
            self.stats.skipped += 1
        elif action in (
            InteractiveAction.EDIT_DESCRIPTION,
            InteractiveAction.EDIT_TAGS,
            InteractiveAction.EDIT_FOLDER,
        ):
            self.stats.edited += 1

    def _undo_last(self) -> None:
        """Undo the last action."""
        if not self.history:
            self._display_message("Nothing to undo", style="yellow")
            return

        last_result = self.history.pop()
        self._restore_state(last_result.bookmark, last_result.original_state)
        self.stats.processed_count -= 1

        # Reverse stats update
        action = last_result.action_taken
        if action == InteractiveAction.ACCEPT_ALL:
            self.stats.accepted_all -= 1
        elif action == InteractiveAction.DESCRIPTION_ONLY:
            self.stats.description_only -= 1
        elif action == InteractiveAction.TAGS_ONLY:
            self.stats.tags_only -= 1
        elif action == InteractiveAction.FOLDER_ONLY:
            self.stats.folder_only -= 1
        elif action == InteractiveAction.SKIP:
            self.stats.skipped -= 1

        self._display_message("Undid last action", style="green")

    def _trigger_save(self, results: List[ProcessedBookmark]) -> None:
        """Trigger save callback."""
        if self._on_save:
            try:
                self._on_save(results)
                self._display_message(
                    f"Progress saved ({len(results)} bookmarks)",
                    style="dim",
                )
            except Exception as e:
                logging.error(f"Error saving progress: {e}")

    # =========================================================================
    # Display Methods
    # =========================================================================

    def _display_welcome(self, count: int) -> None:
        """Display welcome message."""
        if not RICH_AVAILABLE or not self.console:
            print(f"\nInteractive Processing Mode - {count} bookmarks")
            print("=" * 50)
            return

        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Interactive Processing Mode[/bold cyan]\n\n"
                f"[white]Bookmarks to process:[/white] {count}\n"
                f"[white]Auto-accept threshold:[/white] "
                f"{self.confirm_threshold if self.confirm_threshold > 0 else 'None (confirm all)'}\n\n"
                f"[dim]Press 'h' for help at any time[/dim]",
                title="Welcome",
                border_style="cyan",
            )
        )

    def _display_bookmark(
        self,
        index: int,
        total: int,
        bookmark: Bookmark,
        changes: ProposedChanges,
    ) -> None:
        """Display bookmark with proposed changes."""
        if not RICH_AVAILABLE or not self.console:
            self._display_bookmark_plain(index, total, bookmark, changes)
            return

        # Header
        self.console.print()
        self.console.print(
            f"[bold]Bookmark {index + 1}/{total}[/bold] "
            f"[dim]({self.stats.get_progress_percentage():.1f}% complete)[/dim]"
        )

        # URL and title
        title = bookmark.get_effective_title()
        self.console.print(f"[cyan]URL:[/cyan] {bookmark.url[:80]}{'...' if len(bookmark.url) > 80 else ''}")
        self.console.print(f"[cyan]Title:[/cyan] {title[:60]}{'...' if len(title) > 60 else ''}")

        # Confidence indicator
        confidence_style = self._get_confidence_style(changes.overall_confidence)
        self.console.print(
            f"[cyan]Confidence:[/cyan] [{confidence_style}]{changes.overall_confidence:.0%}[/{confidence_style}]"
        )

        self.console.print()

        if self.show_diff:
            self._display_changes_table(changes)
        else:
            self._display_changes_compact(changes)

    def _display_bookmark_plain(
        self,
        index: int,
        total: int,
        bookmark: Bookmark,
        changes: ProposedChanges,
    ) -> None:
        """Display bookmark in plain text (no Rich)."""
        print(f"\n{'=' * 60}")
        print(f"Bookmark {index + 1}/{total}")
        print(f"URL: {bookmark.url}")
        print(f"Title: {bookmark.get_effective_title()}")
        print(f"Confidence: {changes.overall_confidence:.0%}")
        print("-" * 60)

        if changes.has_description_change():
            print("DESCRIPTION:")
            print(f"  Before: {changes.original_description[:100]}...")
            print(f"  After:  {changes.proposed_description[:100]}...")

        if changes.has_tags_change():
            print("TAGS:")
            print(f"  Before: {', '.join(changes.original_tags)}")
            print(f"  After:  {', '.join(changes.proposed_tags)}")

        if changes.has_folder_change():
            print("FOLDER:")
            print(f"  Before: {changes.original_folder}")
            print(f"  After:  {changes.proposed_folder}")

    def _display_changes_table(self, changes: ProposedChanges) -> None:
        """Display changes in a table format."""
        if not RICH_AVAILABLE or not self.console:
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Field", style="cyan", width=12)
        table.add_column("Current", style="dim")
        table.add_column("Proposed", style="green")
        table.add_column("Conf", justify="right", width=6)

        # Description row
        if changes.has_description_change():
            current_desc = (
                changes.original_description[:50] + "..."
                if len(changes.original_description) > 50
                else changes.original_description or "[none]"
            )
            proposed_desc = (
                changes.proposed_description[:50] + "..."
                if len(changes.proposed_description) > 50
                else changes.proposed_description
            )
            conf_style = self._get_confidence_style(changes.description_confidence)
            table.add_row(
                "Description",
                current_desc,
                proposed_desc,
                f"[{conf_style}]{changes.description_confidence:.0%}[/{conf_style}]",
            )

        # Tags row
        if changes.has_tags_change():
            current_tags = ", ".join(changes.original_tags[:3])
            if len(changes.original_tags) > 3:
                current_tags += f" (+{len(changes.original_tags) - 3})"
            proposed_tags = ", ".join(changes.proposed_tags[:3])
            if len(changes.proposed_tags) > 3:
                proposed_tags += f" (+{len(changes.proposed_tags) - 3})"
            conf_style = self._get_confidence_style(changes.tags_confidence)
            table.add_row(
                "Tags",
                current_tags or "[none]",
                proposed_tags,
                f"[{conf_style}]{changes.tags_confidence:.0%}[/{conf_style}]",
            )

        # Folder row
        if changes.has_folder_change():
            conf_style = self._get_confidence_style(changes.folder_confidence)
            table.add_row(
                "Folder",
                changes.original_folder or "[none]",
                changes.proposed_folder,
                f"[{conf_style}]{changes.folder_confidence:.0%}[/{conf_style}]",
            )

        if table.row_count > 0:
            self.console.print(table)
        else:
            self.console.print("[dim]No changes proposed[/dim]")

    def _display_changes_compact(self, changes: ProposedChanges) -> None:
        """Display changes in compact format."""
        if not RICH_AVAILABLE or not self.console:
            return

        if changes.has_description_change():
            self.console.print(
                f"[cyan]Description:[/cyan] [green]{changes.proposed_description[:80]}...[/green]"
            )
        if changes.has_tags_change():
            self.console.print(
                f"[cyan]Tags:[/cyan] [green]{', '.join(changes.proposed_tags)}[/green]"
            )
        if changes.has_folder_change():
            self.console.print(
                f"[cyan]Folder:[/cyan] [green]{changes.proposed_folder}[/green]"
            )

    def _display_help(self) -> None:
        """Display help message."""
        if not RICH_AVAILABLE or not self.console:
            print("\nAvailable actions:")
            print("  a - Accept all changes")
            print("  d - Accept description only")
            print("  t - Accept tags only")
            print("  f - Accept folder only")
            print("  s - Skip (no changes)")
            print("  e - Edit description")
            print("  T - Edit tags")
            print("  F - Edit folder")
            print("  u - Undo last action")
            print("  q - Quit")
            print("  h - Show this help")
            return

        help_text = """
[bold]Available Actions:[/bold]

[cyan]a[/cyan] - Accept all proposed changes
[cyan]d[/cyan] - Accept description change only
[cyan]t[/cyan] - Accept tags change only
[cyan]f[/cyan] - Accept folder change only
[cyan]s[/cyan] - Skip this bookmark (no changes)

[cyan]e[/cyan] - Edit description manually
[cyan]T[/cyan] - Edit tags manually
[cyan]F[/cyan] - Edit folder manually

[cyan]u[/cyan] - Undo last action
[cyan]q[/cyan] - Quit (progress will be saved)
[cyan]h[/cyan] - Show this help
        """
        self.console.print(Panel(help_text.strip(), title="Help", border_style="blue"))

    def _display_summary(self) -> None:
        """Display session summary."""
        if not RICH_AVAILABLE or not self.console:
            print("\nSession Summary:")
            print(f"  Processed: {self.stats.processed_count}/{self.stats.total_bookmarks}")
            print(f"  Accepted all: {self.stats.accepted_all}")
            print(f"  Skipped: {self.stats.skipped}")
            return

        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Session Complete[/bold]\n\n"
                f"Total processed: {self.stats.processed_count}/{self.stats.total_bookmarks}\n"
                f"Accepted all: {self.stats.accepted_all}\n"
                f"Description only: {self.stats.description_only}\n"
                f"Tags only: {self.stats.tags_only}\n"
                f"Folder only: {self.stats.folder_only}\n"
                f"Skipped: {self.stats.skipped}\n"
                f"Edited: {self.stats.edited}\n"
                f"Auto-accepted: {self.stats.auto_accepted}",
                title="Summary",
                border_style="green",
            )
        )

    def _display_quit_message(self, results: List[ProcessedBookmark]) -> None:
        """Display quit message."""
        self._display_message(
            f"Quitting. {len(results)} bookmarks processed.",
            style="yellow",
        )

    def _display_message(self, message: str, style: str = "white") -> None:
        """Display a styled message."""
        if not RICH_AVAILABLE or not self.console:
            print(message)
            return

        self.console.print(f"[{style}]{message}[/{style}]")

    def _get_confidence_style(self, confidence: float) -> str:
        """Get style based on confidence level."""
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.5:
            return "yellow"
        else:
            return "red"

    # =========================================================================
    # Prompt Methods
    # =========================================================================

    def _prompt_action(self) -> InteractiveAction:
        """Prompt user for action."""
        if not RICH_AVAILABLE or not self.console:
            return self._prompt_action_plain()

        choices = "a/d/t/f/s/e/T/F/u/q/h"
        self.console.print()

        try:
            response = Prompt.ask(
                f"[bold]Action[/bold] [{choices}]",
                default="a",
            )
        except (KeyboardInterrupt, EOFError):
            return InteractiveAction.QUIT

        # Map response to action
        action_map = {
            "a": InteractiveAction.ACCEPT_ALL,
            "d": InteractiveAction.DESCRIPTION_ONLY,
            "t": InteractiveAction.TAGS_ONLY,
            "f": InteractiveAction.FOLDER_ONLY,
            "s": InteractiveAction.SKIP,
            "e": InteractiveAction.EDIT_DESCRIPTION,
            "T": InteractiveAction.EDIT_TAGS,
            "F": InteractiveAction.EDIT_FOLDER,
            "u": InteractiveAction.UNDO,
            "q": InteractiveAction.QUIT,
            "h": InteractiveAction.HELP,
        }

        return action_map.get(response.strip(), InteractiveAction.ACCEPT_ALL)

    def _prompt_action_plain(self) -> InteractiveAction:
        """Prompt for action without Rich."""
        print("\n[A]ccept all | [D]escription | [T]ags | [F]older | [S]kip | [Q]uit | [H]elp")

        try:
            response = input("Action [a]: ").strip().lower() or "a"
        except (KeyboardInterrupt, EOFError):
            return InteractiveAction.QUIT

        action_map = {
            "a": InteractiveAction.ACCEPT_ALL,
            "d": InteractiveAction.DESCRIPTION_ONLY,
            "t": InteractiveAction.TAGS_ONLY,
            "f": InteractiveAction.FOLDER_ONLY,
            "s": InteractiveAction.SKIP,
            "q": InteractiveAction.QUIT,
            "h": InteractiveAction.HELP,
            "u": InteractiveAction.UNDO,
        }

        return action_map.get(response, InteractiveAction.ACCEPT_ALL)

    def _prompt_edit_description(self, current: str) -> Optional[str]:
        """Prompt user to edit description."""
        if not RICH_AVAILABLE or not self.console:
            print(f"Current description: {current}")
            try:
                new_desc = input("New description (empty to cancel): ").strip()
                return new_desc if new_desc else None
            except (KeyboardInterrupt, EOFError):
                return None

        self.console.print(f"[dim]Current: {current[:100]}...[/dim]")

        try:
            new_desc = Prompt.ask(
                "New description (empty to cancel)",
                default="",
            )
            return new_desc.strip() if new_desc.strip() else None
        except (KeyboardInterrupt, EOFError):
            return None

    def _prompt_edit_tags(self, current: List[str]) -> Optional[List[str]]:
        """Prompt user to edit tags."""
        if not RICH_AVAILABLE or not self.console:
            print(f"Current tags: {', '.join(current)}")
            try:
                new_tags = input("New tags (comma-separated, empty to cancel): ").strip()
                if not new_tags:
                    return None
                return [t.strip() for t in new_tags.split(",") if t.strip()]
            except (KeyboardInterrupt, EOFError):
                return None

        self.console.print(f"[dim]Current: {', '.join(current)}[/dim]")

        try:
            new_tags = Prompt.ask(
                "New tags (comma-separated, empty to cancel)",
                default="",
            )
            if not new_tags.strip():
                return None
            return [t.strip() for t in new_tags.split(",") if t.strip()]
        except (KeyboardInterrupt, EOFError):
            return None

    def _prompt_edit_folder(self, current: str) -> Optional[str]:
        """Prompt user to edit folder."""
        if not RICH_AVAILABLE or not self.console:
            print(f"Current folder: {current}")
            try:
                new_folder = input("New folder (empty to cancel): ").strip()
                return new_folder if new_folder else None
            except (KeyboardInterrupt, EOFError):
                return None

        self.console.print(f"[dim]Current: {current}[/dim]")

        try:
            new_folder = Prompt.ask(
                "New folder (empty to cancel)",
                default="",
            )
            return new_folder.strip() if new_folder.strip() else None
        except (KeyboardInterrupt, EOFError):
            return None


__all__ = [
    "InteractiveProcessor",
    "InteractiveAction",
    "ProposedChanges",
    "ProcessedBookmark",
    "InteractiveSessionStats",
]
