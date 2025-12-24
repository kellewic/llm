"""Test cases for server.py functions: scrape_websites and extract_scraped_info."""

import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the functions we want to test
from server import SCRAPE_DIR, extract_scraped_info, scrape_websites


class TestScrapeWebsites:
    """Test cases for the scrape_websites function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_firecrawl_app(self):
        """Mock FirecrawlApp."""
        with patch("server.FirecrawlApp") as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app
            yield mock_app

    @pytest.fixture
    def mock_env_api_key(self):
        """Mock environment variable for API key."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-api-key"}):
            yield

    def test_scrape_websites_successful_scrape(self, temp_dir, mock_firecrawl_app, mock_env_api_key):
        """Test successful scraping of websites."""
        # Setup
        with patch("server.SCRAPE_DIR", temp_dir):
            websites = {
                "test_provider": "https://example.com",
            }
            formats = ["markdown", "html"]

            # Mock successful scrape result
            mock_scrape_result = Mock()
            mock_scrape_result.model_dump.return_value = {
                "success": True,
                "title": "Test Page",
                "description": "Test Description",
                "markdown": "# Test Markdown Content",
                "html": "<html><body>Test HTML Content</body></html>",
            }
            mock_firecrawl_app.scrape_url.return_value = mock_scrape_result

            # Execute
            result = scrape_websites(websites, formats=formats)

            # Assert
            assert result == ["test_provider"]
            assert len(result) == 1

            # Check metadata file was created
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            assert os.path.exists(metadata_file)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert "test_provider" in metadata
            assert metadata["test_provider"]["success"] is True
            assert metadata["test_provider"]["title"] == "Test Page"
            assert metadata["test_provider"]["description"] == "Test Description"
            assert metadata["test_provider"]["url"] == "https://example.com"
            assert metadata["test_provider"]["domain"] == "example.com"
            assert "content_files" in metadata["test_provider"]
            assert "markdown" in metadata["test_provider"]["content_files"]
            assert "html" in metadata["test_provider"]["content_files"]

            # Check content files were created
            markdown_file = os.path.join(temp_dir, "test_provider_markdown.txt")
            html_file = os.path.join(temp_dir, "test_provider_html.txt")
            assert os.path.exists(markdown_file)
            assert os.path.exists(html_file)

            with open(markdown_file, "r") as f:
                assert f.read() == "# Test Markdown Content"

            with open(html_file, "r") as f:
                assert f.read() == "<html><body>Test HTML Content</body></html>"

    def test_scrape_websites_failed_scrape(self, temp_dir, mock_firecrawl_app, mock_env_api_key):
        """Test handling of failed scrape."""
        with patch("server.SCRAPE_DIR", temp_dir):
            websites = {
                "failed_provider": "https://example.com",
            }

            # Mock failed scrape result
            mock_scrape_result = Mock()
            mock_scrape_result.model_dump.return_value = {
                "success": False,
                "error": "Failed to scrape",
            }
            mock_firecrawl_app.scrape_url.return_value = mock_scrape_result

            # Execute
            result = scrape_websites(websites)

            # Assert
            assert result == []  # No successful scrapes
            assert len(result) == 0

            # Check metadata file was created with failure entry
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            assert os.path.exists(metadata_file)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert "failed_provider" in metadata
            assert metadata["failed_provider"]["success"] is False
            assert metadata["failed_provider"]["content_files"] == {}

    def test_scrape_websites_loads_existing_metadata(self, temp_dir, mock_firecrawl_app, mock_env_api_key):
        """Test that existing metadata is loaded and preserved."""
        with patch("server.SCRAPE_DIR", temp_dir):
            # Create existing metadata file
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            existing_metadata = {
                "existing_provider": {
                    "provider_name": "existing_provider",
                    "url": "https://existing.com",
                    "domain": "existing.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": ["markdown"],
                    "success": True,
                    "content_files": {"markdown": "existing_provider_markdown.txt"},
                    "title": "Existing",
                    "description": "Existing Description",
                }
            }
            with open(metadata_file, "w") as f:
                json.dump(existing_metadata, f)

            websites = {
                "new_provider": "https://new.com",
            }

            # Mock successful scrape result
            mock_scrape_result = Mock()
            mock_scrape_result.model_dump.return_value = {
                "success": True,
                "title": "New Page",
                "description": "New Description",
                "markdown": "# New Content",
                "html": "<html>New HTML</html>",
            }
            mock_firecrawl_app.scrape_url.return_value = mock_scrape_result

            # Execute
            result = scrape_websites(websites)

            # Assert
            assert result == ["new_provider"]

            # Check both providers are in metadata
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert "existing_provider" in metadata
            assert "new_provider" in metadata
            assert metadata["existing_provider"]["title"] == "Existing"
            assert metadata["new_provider"]["title"] == "New Page"

    def test_scrape_websites_empty_metadata_file(self, temp_dir, mock_firecrawl_app, mock_env_api_key):
        """Test handling of empty metadata file."""
        with patch("server.SCRAPE_DIR", temp_dir):
            # Create empty metadata file
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump({}, f)

            websites = {
                "test_provider": "https://example.com",
            }

            mock_scrape_result = Mock()
            mock_scrape_result.model_dump.return_value = {
                "success": True,
                "title": "Test",
                "description": "Test",
                "markdown": "# Test",
                "html": "<html>Test</html>",
            }
            mock_firecrawl_app.scrape_url.return_value = mock_scrape_result

            # Execute
            result = scrape_websites(websites)

            # Assert
            assert result == ["test_provider"]

    def test_scrape_websites_multiple_websites(self, temp_dir, mock_firecrawl_app, mock_env_api_key):
        """Test scraping multiple websites."""
        with patch("server.SCRAPE_DIR", temp_dir):
            websites = {
                "provider1": "https://example1.com",
                "provider2": "https://example2.com",
            }

            # Mock successful scrape results
            def mock_scrape_side_effect(url, formats):
                mock_result = Mock()
                if "example1.com" in url:
                    mock_result.model_dump.return_value = {
                        "success": True,
                        "title": "Provider 1",
                        "description": "Description 1",
                        "markdown": "# Provider 1",
                        "html": "<html>Provider 1</html>",
                    }
                else:
                    mock_result.model_dump.return_value = {
                        "success": True,
                        "title": "Provider 2",
                        "description": "Description 2",
                        "markdown": "# Provider 2",
                        "html": "<html>Provider 2</html>",
                    }
                return mock_result

            mock_firecrawl_app.scrape_url.side_effect = mock_scrape_side_effect

            # Execute
            result = scrape_websites(websites)

            # Assert
            assert len(result) == 2
            assert "provider1" in result
            assert "provider2" in result

            # Check metadata contains both
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert len(metadata) == 2
            assert "provider1" in metadata
            assert "provider2" in metadata

    def test_scrape_websites_missing_api_key(self, temp_dir):
        """Test error when API key is missing."""
        with patch("server.SCRAPE_DIR", temp_dir):
            with patch.dict(os.environ, {}, clear=True):
                websites = {"test": "https://example.com"}

                with pytest.raises(ValueError, match="API key must be provided"):
                    scrape_websites(websites)

    def test_scrape_websites_custom_formats(self, temp_dir, mock_firecrawl_app, mock_env_api_key):
        """Test scraping with custom formats."""
        with patch("server.SCRAPE_DIR", temp_dir):
            websites = {"test_provider": "https://example.com"}
            formats = ["markdown"]  # Only markdown

            mock_scrape_result = Mock()
            mock_scrape_result.model_dump.return_value = {
                "success": True,
                "title": "Test",
                "description": "Test",
                "markdown": "# Test",
            }
            mock_firecrawl_app.scrape_url.return_value = mock_scrape_result

            # Execute
            result = scrape_websites(websites, formats=formats)

            # Assert
            assert result == ["test_provider"]

            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert "markdown" in metadata["test_provider"]["content_files"]
            assert "html" not in metadata["test_provider"]["content_files"]


class TestExtractScrapedInfo:
    """Test cases for the extract_scraped_info function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_extract_scraped_info_by_provider_name(self, temp_dir):
        """Test extracting info by provider name."""
        with patch("server.SCRAPE_DIR", temp_dir):
            # Setup metadata and content files
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            metadata = {
                "test_provider": {
                    "provider_name": "test_provider",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": ["markdown", "html"],
                    "success": True,
                    "content_files": {
                        "markdown": "test_provider_markdown.txt",
                        "html": "test_provider_html.txt",
                    },
                    "title": "Test Page",
                    "description": "Test Description",
                }
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Create content files
            with open(os.path.join(temp_dir, "test_provider_markdown.txt"), "w") as f:
                f.write("# Test Markdown")
            with open(os.path.join(temp_dir, "test_provider_html.txt"), "w") as f:
                f.write("<html>Test HTML</html>")

            # Execute
            result = extract_scraped_info("test_provider")

            # Assert
            assert result is not None
            result_dict = json.loads(result)
            assert result_dict["provider_name"] == "test_provider"
            assert result_dict["url"] == "https://example.com"
            assert "content" in result_dict
            assert result_dict["content"]["markdown"] == "# Test Markdown"
            assert result_dict["content"]["html"] == "<html>Test HTML</html>"

    def test_extract_scraped_info_by_url(self, temp_dir):
        """Test extracting info by URL."""
        with patch("server.SCRAPE_DIR", temp_dir):
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            metadata = {
                "test_provider": {
                    "provider_name": "test_provider",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": ["markdown"],
                    "success": True,
                    "content_files": {"markdown": "test_provider_markdown.txt"},
                    "title": "Test",
                    "description": "Test",
                }
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            with open(os.path.join(temp_dir, "test_provider_markdown.txt"), "w") as f:
                f.write("# Test")

            # Execute
            result = extract_scraped_info("https://example.com")

            # Assert
            result_dict = json.loads(result)
            assert result_dict["url"] == "https://example.com"

    def test_extract_scraped_info_by_domain(self, temp_dir):
        """Test extracting info by domain."""
        with patch("server.SCRAPE_DIR", temp_dir):
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            metadata = {
                "test_provider": {
                    "provider_name": "test_provider",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": ["markdown"],
                    "success": True,
                    "content_files": {"markdown": "test_provider_markdown.txt"},
                    "title": "Test",
                    "description": "Test",
                }
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            with open(os.path.join(temp_dir, "test_provider_markdown.txt"), "w") as f:
                f.write("# Test")

            # Execute
            result = extract_scraped_info("example.com")

            # Assert
            result_dict = json.loads(result)
            assert result_dict["domain"] == "example.com"

    def test_extract_scraped_info_no_match(self, temp_dir):
        """Test extracting info with no matching identifier."""
        with patch("server.SCRAPE_DIR", temp_dir):
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            metadata = {
                "test_provider": {
                    "provider_name": "test_provider",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": ["markdown"],
                    "success": True,
                    "content_files": {"markdown": "test_provider_markdown.txt"},
                    "title": "Test",
                    "description": "Test",
                }
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Execute
            result = extract_scraped_info("nonexistent")

            # Assert
            assert "There's no saved information related to identifier 'nonexistent'" in result
            assert not result.startswith("{")  # Should be plain text, not JSON

    def test_extract_scraped_info_metadata_file_not_found(self, temp_dir):
        """Test handling when metadata file doesn't exist."""
        with patch("server.SCRAPE_DIR", temp_dir):
            # Don't create metadata file

            # Execute
            result = extract_scraped_info("test_provider")

            # Assert
            assert "There's no saved information related to identifier 'test_provider'" in result

    def test_extract_scraped_info_missing_content_file(self, temp_dir):
        """Test handling when content file is missing."""
        with patch("server.SCRAPE_DIR", temp_dir):
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            metadata = {
                "test_provider": {
                    "provider_name": "test_provider",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": ["markdown"],
                    "success": True,
                    "content_files": {"markdown": "missing_file.txt"},
                    "title": "Test",
                    "description": "Test",
                }
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Don't create the content file

            # Execute
            result = extract_scraped_info("test_provider")

            # Assert - should still return metadata but with empty content
            result_dict = json.loads(result)
            assert result_dict["provider_name"] == "test_provider"
            assert result_dict["content"]["markdown"] == ""

    def test_extract_scraped_info_no_content_files(self, temp_dir):
        """Test extracting info when there are no content files."""
        with patch("server.SCRAPE_DIR", temp_dir):
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            metadata = {
                "test_provider": {
                    "provider_name": "test_provider",
                    "url": "https://example.com",
                    "domain": "example.com",
                    "scraped_at": "2024-01-01T00:00:00",
                    "formats": [],
                    "success": True,
                    "content_files": {},
                    "title": "Test",
                    "description": "Test",
                }
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Execute
            result = extract_scraped_info("test_provider")

            # Assert - should return metadata without content field
            result_dict = json.loads(result)
            assert result_dict["provider_name"] == "test_provider"
            assert "content" not in result_dict

    def test_extract_scraped_info_invalid_json(self, temp_dir):
        """Test handling of invalid JSON in metadata file."""
        with patch("server.SCRAPE_DIR", temp_dir):
            metadata_file = os.path.join(temp_dir, "scraped_metadata.json")
            # Write invalid JSON
            with open(metadata_file, "w") as f:
                f.write("invalid json {")

            # Execute
            result = extract_scraped_info("test_provider")

            # Assert
            assert "There's no saved information related to identifier 'test_provider'" in result
