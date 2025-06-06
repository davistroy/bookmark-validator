"""
Tests for AI-powered folder generation functionality.
"""

import pytest
from datetime import datetime
from typing import List, Dict

from bookmark_processor.core.folder_generator import (
    AIFolderGenerator, FolderNode, FolderGenerationResult
)
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.content_analyzer import ContentData
from bookmark_processor.core.ai_processor import AIProcessingResult


class TestFolderNode:
    """Test FolderNode data structure"""
    
    def test_folder_node_initialization(self):
        """Test basic folder node creation"""
        node = FolderNode(name="Development", path="Development")
        
        assert node.name == "Development"
        assert node.path == "Development"
        assert node.parent is None
        assert node.children == []
        assert node.bookmarks == []
    
    def test_add_bookmark(self):
        """Test adding bookmarks to folder"""
        node = FolderNode(name="Python", path="Development/Python")
        bookmark = Bookmark(
            url="https://python.org",
            title="Python.org",
            created=datetime.now()
        )
        
        node.add_bookmark(bookmark)
        
        assert len(node.bookmarks) == 1
        assert node.bookmarks[0] == bookmark
    
    def test_add_child(self):
        """Test adding child folders"""
        parent = FolderNode(name="Development", path="Development")
        child = FolderNode(name="Python", path="Development/Python")
        
        parent.add_child(child)
        
        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
    
    def test_get_full_path(self):
        """Test getting full folder path"""
        root = FolderNode(name="root", path="")
        dev = FolderNode(name="Development", path="Development")
        python = FolderNode(name="Python", path="Development/Python")
        
        root.add_child(dev)
        dev.add_child(python)
        
        assert python.get_full_path() == "Development/Python"
        assert dev.get_full_path() == "Development"
        assert root.get_full_path() == "root"
    
    def test_get_bookmark_count(self):
        """Test recursive bookmark counting"""
        root = FolderNode(name="root", path="")
        dev = FolderNode(name="Development", path="Development")
        python = FolderNode(name="Python", path="Development/Python")
        
        root.add_child(dev)
        dev.add_child(python)
        
        # Add bookmarks at different levels
        root.add_bookmark(Bookmark(url="https://example.com", title="Example", created=datetime.now()))
        dev.add_bookmark(Bookmark(url="https://dev.com", title="Dev", created=datetime.now()))
        dev.add_bookmark(Bookmark(url="https://code.com", title="Code", created=datetime.now()))
        python.add_bookmark(Bookmark(url="https://python.org", title="Python", created=datetime.now()))
        
        assert python.get_bookmark_count() == 1
        assert dev.get_bookmark_count() == 3  # 2 own + 1 from child
        assert root.get_bookmark_count() == 4  # 1 own + 3 from children


class TestAIFolderGenerator:
    """Test AI folder generator"""
    
    @pytest.fixture
    def sample_bookmarks(self) -> List[Bookmark]:
        """Create sample bookmarks for testing"""
        return [
            Bookmark(
                url="https://github.com/python/cpython",
                title="Python Source Code",
                note="Official Python implementation",
                tags="python, programming",
                folder="Development",
                created=datetime.now()
            ),
            Bookmark(
                url="https://react.dev",
                title="React Documentation",
                note="Official React docs",
                tags="react, javascript, frontend",
                folder="Web Development",
                created=datetime.now()
            ),
            Bookmark(
                url="https://arxiv.org/abs/2021.12345",
                title="Deep Learning Paper",
                note="Research on neural networks",
                tags="ai, research",
                folder="Research/AI",
                created=datetime.now()
            ),
            Bookmark(
                url="https://medium.com/tech-blog",
                title="Tech Blog Post",
                note="Article about software development",
                tags="blog, technology",
                folder="Reading",
                created=datetime.now()
            ),
            Bookmark(
                url="https://stackoverflow.com/questions/12345",
                title="Python Question",
                note="How to use decorators",
                tags="python, stackoverflow",
                folder="Development/Python",
                created=datetime.now()
            )
        ]
    
    @pytest.fixture
    def content_data_map(self, sample_bookmarks) -> Dict[str, ContentData]:
        """Create sample content data"""
        content_map = {}
        
        for bookmark in sample_bookmarks:
            content = ContentData(url=bookmark.url)
            content.title = bookmark.title
            content.meta_description = f"Content about {bookmark.title}"
            content.content_summary = bookmark.note
            content_map[bookmark.url] = content
        
        return content_map
    
    @pytest.fixture
    def folder_generator(self) -> AIFolderGenerator:
        """Create folder generator instance"""
        return AIFolderGenerator(max_bookmarks_per_folder=20)
    
    def test_folder_generator_initialization(self, folder_generator):
        """Test folder generator initialization"""
        assert folder_generator.max_bookmarks_per_folder == 20
        assert folder_generator.ai_processor is not None
        assert len(folder_generator.category_patterns) > 0
    
    def test_extract_domain(self, folder_generator):
        """Test domain extraction"""
        assert folder_generator._extract_domain("https://github.com/user/repo") == "github.com"
        assert folder_generator._extract_domain("https://www.example.com/page") == "example.com"
        assert folder_generator._extract_domain("http://subdomain.site.com") == "subdomain.site.com"
        assert folder_generator._extract_domain("invalid-url") == ""
    
    def test_determine_category(self, folder_generator, sample_bookmarks, content_data_map):
        """Test category determination"""
        # Test Python bookmark
        python_bookmark = sample_bookmarks[0]
        content = content_data_map[python_bookmark.url]
        category, subcategory = folder_generator._determine_category(
            python_bookmark, content, None, "Development"
        )
        assert category == "Development"
        assert subcategory in ["Backend", "General"]
        
        # Test AI research bookmark
        ai_bookmark = sample_bookmarks[2]
        content = content_data_map[ai_bookmark.url]
        category, subcategory = folder_generator._determine_category(
            ai_bookmark, content, None, "Research/AI"
        )
        assert category == "AI & Machine Learning"
        assert subcategory in ["Research", "General"]
    
    def test_generate_folder_structure_basic(self, folder_generator, sample_bookmarks, content_data_map):
        """Test basic folder structure generation"""
        result = folder_generator.generate_folder_structure(
            sample_bookmarks,
            content_data_map=content_data_map,
            original_folders_map={b.url: b.folder for b in sample_bookmarks}
        )
        
        assert isinstance(result, FolderGenerationResult)
        assert result.total_folders > 0
        assert len(result.folder_assignments) == len(sample_bookmarks)
        assert all(url in result.folder_assignments for url in [b.url for b in sample_bookmarks])
    
    def test_folder_size_limits(self, folder_generator):
        """Test folder size limit enforcement"""
        # Create many bookmarks that would go in same category
        many_bookmarks = []
        for i in range(30):  # More than max_bookmarks_per_folder
            bookmark = Bookmark(
                url=f"https://github.com/project{i}",
                title=f"GitHub Project {i}",
                tags="github, development",
                created=datetime.now()
            )
            many_bookmarks.append(bookmark)
        
        result = folder_generator.generate_folder_structure(many_bookmarks)
        
        # Check that no folder has more than max bookmarks
        for folder_path, count in result.folder_stats.items():
            assert count <= folder_generator.max_bookmarks_per_folder
    
    def test_hierarchical_structure(self, folder_generator, sample_bookmarks, content_data_map):
        """Test hierarchical folder structure creation"""
        result = folder_generator.generate_folder_structure(
            sample_bookmarks,
            content_data_map=content_data_map
        )
        
        # Check for hierarchical paths
        has_hierarchical = any('/' in path for path in result.folder_assignments.values())
        assert result.max_depth >= 1  # At least some hierarchy
    
    def test_fallback_category(self, folder_generator):
        """Test fallback category assignment"""
        bookmark = Bookmark(
            url="https://random-site.com",
            title="Random Site",
            created=datetime.now()
        )
        
        # No matching patterns, no original folder
        category = folder_generator._fallback_category(bookmark, "")
        assert category == "Uncategorized"
        
        # With original folder hint
        category = folder_generator._fallback_category(bookmark, "MySpecialFolder/Subfolder")
        assert category == "Myspecialfolder"  # Extracted and formatted
    
    def test_folder_report_generation(self, folder_generator, sample_bookmarks):
        """Test folder structure report generation"""
        result = folder_generator.generate_folder_structure(sample_bookmarks)
        report = folder_generator.get_folder_report(result)
        
        assert "AI-Generated Folder Structure Report" in report
        assert f"Total Folders: {result.total_folders}" in report
        assert f"Maximum Depth: {result.max_depth}" in report
        assert "Folder Hierarchy:" in report
    
    def test_group_similar_bookmarks(self, folder_generator):
        """Test bookmark grouping by similarity"""
        bookmarks = [
            Bookmark(url="https://github.com/repo1", title="Repo 1", created=datetime.now()),
            Bookmark(url="https://github.com/repo2", title="Repo 2", created=datetime.now()),
            Bookmark(url="https://github.com/repo3", title="Repo 3", created=datetime.now()),
            Bookmark(url="https://stackoverflow.com/q1", title="Question 1", created=datetime.now()),
            Bookmark(url="https://stackoverflow.com/q2", title="Question 2", created=datetime.now()),
            Bookmark(url="https://example.com", title="Example", created=datetime.now()),
        ]
        
        groups = folder_generator._group_similar_bookmarks(bookmarks)
        
        assert "Github" in groups or "github" in [g.lower() for g in groups]
        assert len(groups) >= 2  # At least github and something else
    
    def test_empty_bookmarks_handling(self, folder_generator):
        """Test handling of empty bookmark list"""
        result = folder_generator.generate_folder_structure([])
        
        assert result.total_folders == 0
        assert len(result.folder_assignments) == 0
        assert result.max_depth == 0


class TestIntegrationWithPipeline:
    """Test integration with main processing pipeline"""
    
    def test_folder_generation_result_serialization(self):
        """Test that results can be serialized for checkpointing"""
        # Create test data
        folder_generator = AIFolderGenerator(max_bookmarks_per_folder=20)
        sample_bookmarks = [
            Bookmark(
                url="https://github.com/test",
                title="Test Repo",
                created=datetime.now()
            )
        ]
        
        result = folder_generator.generate_folder_structure(sample_bookmarks)
        
        # Test serialization
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'folder_assignments' in result_dict
        assert 'total_folders' in result_dict
        assert 'max_depth' in result_dict
        assert 'folder_stats' in result_dict
        assert 'processing_time' in result_dict
        
        # All URLs should be in assignments
        for bookmark in sample_bookmarks:
            assert bookmark.url in result_dict['folder_assignments']