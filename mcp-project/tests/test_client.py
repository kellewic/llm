"""Test cases for client.py classes and functions."""

import asyncio
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from client import (
    ChatSession,
    Configuration,
    DataExtractor,
    Server,
)


class TestConfiguration:
    """Test cases for the Configuration class."""

    def test_load_env(self):
        """Test loading environment variables."""
        config = Configuration()
        assert hasattr(config, "api_key")
        assert hasattr(config, "model_name")
        assert hasattr(config, "api_url")

    def test_load_config_success(self, tmp_path):
        """Test successful loading of configuration file."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "mcpServers": {
                "test_server": {
                    "command": "python",
                    "args": ["test.py"],
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        result = Configuration.load_config(str(config_file))
        assert "mcpServers" in result
        assert "test_server" in result["mcpServers"]

    def test_load_config_file_not_found(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            Configuration.load_config("nonexistent.json")

    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON configuration file."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            Configuration.load_config(str(config_file))

    def test_load_config_missing_mcp_servers(self, tmp_path):
        """Test loading configuration file missing mcpServers field."""
        config_file = tmp_path / "invalid_config.json"
        config_file.write_text(json.dumps({"other": "data"}))

        with pytest.raises(ValueError, match="mcpServers"):
            Configuration.load_config(str(config_file))

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_API_MODEL": "test-model"})
    def test_anthropic_api_key_property(self):
        """Test anthropic_api_key property."""
        config = Configuration()
        assert config.anthropic_api_key == "test-key"

    def test_anthropic_api_key_missing(self):
        """Test anthropic_api_key property when key is missing."""
        # Create a config and manually set api_key to None to test the property
        config = Configuration()
        # Save original and set to None
        original_key = config.api_key
        config.api_key = None
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                _ = config.anthropic_api_key
        finally:
            # Restore original
            config.api_key = original_key

    @patch.dict(os.environ, {"ANTHROPIC_API_MODEL": "test-model"})
    def test_anthropic_model_property(self):
        """Test anthropic_model property."""
        config = Configuration()
        assert config.anthropic_model == "test-model"

    @patch.dict(os.environ, {"ANTHROPIC_API_URL": "https://custom.url"})
    def test_anthropic_api_url_property(self):
        """Test anthropic_api_url property with custom URL."""
        config = Configuration()
        assert config.anthropic_api_url == "https://custom.url"

    def test_anthropic_api_url_none(self):
        """Test anthropic_api_url property when URL is not set."""
        # Create a config and manually set api_url to None to test the property
        config = Configuration()
        # Save original and set to None
        original_url = config.api_url
        config.api_url = None
        try:
            assert config.anthropic_api_url is None
        finally:
            # Restore original
            config.api_url = original_url

    def test_load_config_generic_exception(self, tmp_path):
        """Test load_config with generic exception."""
        config_file = tmp_path / "test.json"
        config_file.write_text("{}")
        
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(ValueError, match="Error loading configuration"):
                Configuration.load_config(str(config_file))


class TestServer:
    """Test cases for the Server class."""

    @pytest.fixture
    def server_config(self):
        """Create a sample server configuration."""
        return {
            "command": "python",
            "args": ["test.py"],
            "env": {"TEST_VAR": "test_value"},
        }

    @pytest.fixture
    def server(self, server_config):
        """Create a Server instance."""
        return Server("test_server", server_config)

    @pytest.mark.asyncio
    async def test_initialize_success(self, server):
        """Test successful server initialization."""
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_transport = (mock_read, mock_write)
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        # Mock the async context managers properly
        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__ = AsyncMock(return_value=mock_transport)
        mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch("client.stdio_client", return_value=mock_stdio_context):
            with patch("client.ClientSession", return_value=mock_session_context):

                await server.initialize()

                assert server.session is not None
                mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_success(self, server):
        """Test listing tools from server."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.inputSchema = {"type": "object"}

        mock_tools_response = Mock()
        mock_tools_response.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        server.session = mock_session

        tools = await server.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "Test tool description"
        assert tools[0]["input_schema"] == {"type": "object"}

    @pytest.mark.asyncio
    async def test_list_tools_no_session(self, server):
        """Test listing tools when session is not initialized."""
        server.session = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.list_tools()

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, server):
        """Test successful tool execution."""
        mock_result = Mock()
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        server.session = mock_session

        result = await server.execute_tool("test_tool", {"arg": "value"})

        assert result == mock_result
        mock_session.call_tool.assert_called_once_with(
            name="test_tool", arguments={"arg": "value"}, read_timeout_seconds=timedelta(seconds=60)
        )

    @pytest.mark.asyncio
    async def test_execute_tool_with_retries(self, server):
        """Test tool execution with retry mechanism."""
        mock_result = Mock()
        mock_session = AsyncMock()
        # First call fails, second succeeds
        mock_session.call_tool = AsyncMock(side_effect=[Exception("Error"), mock_result])
        server.session = mock_session

        result = await server.execute_tool("test_tool", {"arg": "value"}, retries=1, delay=0.01)

        assert result == mock_result
        assert mock_session.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tool_all_retries_fail(self, server):
        """Test tool execution when all retries fail."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Persistent error"))
        server.session = mock_session

        with pytest.raises(Exception, match="Persistent error"):
            await server.execute_tool("test_tool", {"arg": "value"}, retries=2, delay=0.01)

        assert mock_session.call_tool.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_execute_tool_no_session(self, server):
        """Test tool execution when session is not initialized."""
        server.session = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.execute_tool("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_cleanup(self, server):
        """Test server cleanup."""
        mock_exit_stack = AsyncMock()
        server.exit_stack = mock_exit_stack
        server.session = Mock()

        await server.cleanup()

        mock_exit_stack.aclose.assert_called_once()
        assert server.session is None

    @pytest.mark.asyncio
    async def test_initialize_failure(self, server):
        """Test server initialization failure."""
        # Mock stdio_client to raise an exception
        with patch("client.stdio_client", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await server.initialize()

    @pytest.mark.asyncio
    async def test_initialize_command_none(self, server_config):
        """Test server initialization with None command."""
        server_config["command"] = None
        server = Server("test_server", server_config)
        
        with pytest.raises(ValueError, match="cannot be None"):
            await server.initialize()

    @pytest.mark.asyncio
    async def test_initialize_with_env(self, server_config):
        """Test server initialization with environment variables."""
        server_config["env"] = {"CUSTOM_VAR": "custom_value"}
        server = Server("test_server", server_config)
        
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_transport = (mock_read, mock_write)
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        mock_stdio_context = AsyncMock()
        mock_stdio_context.__aenter__ = AsyncMock(return_value=mock_transport)
        mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch("client.stdio_client", return_value=mock_stdio_context):
            with patch("client.ClientSession", return_value=mock_session_context):
                await server.initialize()
                assert server.session is not None

    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self, server):
        """Test server cleanup with exception."""
        mock_exit_stack = AsyncMock()
        mock_exit_stack.aclose = AsyncMock(side_effect=Exception("Cleanup error"))
        server.exit_stack = mock_exit_stack
        server.session = Mock()

        # Should not raise, just log the error
        await server.cleanup()

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_error(self, server):
        """Test execute_tool when last_exception is None (shouldn't happen but test the branch)."""
        mock_session = AsyncMock()
        # Mock to return None somehow (edge case)
        mock_session.call_tool = AsyncMock(return_value=None)
        server.session = mock_session

        # This shouldn't raise RuntimeError in normal flow, but let's test the branch
        # by making it fail in an unexpected way
        result = await server.execute_tool("test_tool", {"arg": "value"}, retries=0)
        # Should return the result, not raise RuntimeError
        assert result is None


class TestDataExtractor:
    """Test cases for the DataExtractor class."""

    @pytest.fixture
    def mock_sqlite_server(self):
        """Create a mock SQLite server."""
        server = AsyncMock()
        server.execute_tool = AsyncMock()
        return server

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        client = Mock()
        client.messages = Mock()
        return client

    @pytest.fixture
    def data_extractor(self, mock_sqlite_server, mock_anthropic_client):
        """Create a DataExtractor instance."""
        return DataExtractor(mock_sqlite_server, mock_anthropic_client, "test-model")

    @pytest.mark.asyncio
    async def test_setup_data_tables(self, data_extractor, mock_sqlite_server):
        """Test setting up data tables."""
        await data_extractor.setup_data_tables()

        mock_sqlite_server.execute_tool.assert_called_once()
        call_args = mock_sqlite_server.execute_tool.call_args
        assert call_args[0][0] == "write_query"
        assert "CREATE TABLE IF NOT EXISTS pricing_plans" in call_args[0][1]["query"]

    @pytest.mark.asyncio
    async def test_get_structured_extraction(self, data_extractor, mock_anthropic_client):
        """Test structured data extraction from LLM."""
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = '{"company_name": "Test Co", "plans": []}'

        mock_response = Mock()
        mock_response.content = [mock_content]

        mock_anthropic_client.messages.create.return_value = mock_response

        result = await data_extractor._get_structured_extraction("test prompt")

        assert "company_name" in result
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_and_store_data(self, data_extractor, mock_sqlite_server, mock_anthropic_client):
        """Test extracting and storing pricing data."""
        # Mock the extraction response
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = json.dumps({
            "company_name": "Test Company",
            "plans": [
                {
                    "plan_name": "Test Plan",
                    "input_tokens": 0.1,
                    "output_tokens": 0.2,
                    "currency": "USD",
                    "billing_period": "monthly",
                    "features": ["feature1"],
                    "limitations": "None",
                }
            ],
        })

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        await data_extractor.extract_and_store_data("test query", "test response")

        # Verify that execute_tool was called to insert the plan
        assert mock_sqlite_server.execute_tool.called
        call_args = mock_sqlite_server.execute_tool.call_args
        assert call_args[0][0] == "write_query"
        assert "INSERT INTO pricing_plans" in call_args[0][1]["query"]
        assert "Test Company" in call_args[0][1]["query"]
        assert "Test Plan" in call_args[0][1]["query"]

    @pytest.mark.asyncio
    async def test_extract_and_store_data_with_none_values(self, data_extractor, mock_sqlite_server, mock_anthropic_client):
        """Test extracting and storing data with None values."""
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = json.dumps({
            "company_name": "Test Company",
            "plans": [
                {
                    "plan_name": "Test Plan",
                    "input_tokens": None,
                    "output_tokens": None,
                    "currency": "USD",
                }
            ],
        })

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        await data_extractor.extract_and_store_data("test query", "test response")

        # Verify that NULL is used for None values
        call_args = mock_sqlite_server.execute_tool.call_args
        assert "NULL" in call_args[0][1]["query"]

    @pytest.mark.asyncio
    async def test_extract_and_store_data_json_error(self, data_extractor, mock_sqlite_server, mock_anthropic_client):
        """Test extract_and_store_data with invalid JSON response."""
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = "not valid json"

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Should handle JSON decode error gracefully
        await data_extractor.extract_and_store_data("test query", "test response")

    @pytest.mark.asyncio
    async def test_extract_and_store_data_no_plans(self, data_extractor, mock_sqlite_server, mock_anthropic_client):
        """Test extract_and_store_data with no plans in response."""
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = json.dumps({
            "company_name": "Test Company",
            "plans": [],
        })

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        await data_extractor.extract_and_store_data("test query", "test response")

        # Should not call execute_tool if no plans
        assert not mock_sqlite_server.execute_tool.called

    @pytest.mark.asyncio
    async def test_get_structured_extraction_error(self, data_extractor, mock_anthropic_client):
        """Test _get_structured_extraction with error."""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        result = await data_extractor._get_structured_extraction("test prompt")
        assert "error" in result
        assert "extraction failed" in result

    @pytest.mark.asyncio
    async def test_get_structured_extraction_no_text_content(self, data_extractor, mock_anthropic_client):
        """Test _get_structured_extraction with no text content."""
        mock_content = Mock()
        mock_content.type = "image"  # Not text

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = await data_extractor._get_structured_extraction("test prompt")
        assert result == ""  # Should return empty string

    @pytest.mark.asyncio
    async def test_setup_data_tables_error(self, data_extractor, mock_sqlite_server):
        """Test setup_data_tables with error."""
        mock_sqlite_server.execute_tool = AsyncMock(side_effect=Exception("DB Error"))

        await data_extractor.setup_data_tables()
        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_extract_and_store_data_insert_error(self, data_extractor, mock_sqlite_server, mock_anthropic_client):
        """Test extract_and_store_data with error during insert."""
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = json.dumps({
            "company_name": "Test Company",
            "plans": [
                {
                    "plan_name": "Test Plan",
                    "input_tokens": 0.1,
                    "output_tokens": 0.2,
                }
            ],
        })

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Mock execute_tool to fail on insert
        mock_sqlite_server.execute_tool = AsyncMock(side_effect=Exception("Insert error"))

        await data_extractor.extract_and_store_data("test query", "test response")
        # Should handle error gracefully


class TestChatSession:
    """Test cases for the ChatSession class."""

    @pytest.fixture
    def mock_servers(self):
        """Create mock servers."""
        server1 = AsyncMock()
        server1.name = "test_server"
        server1.list_tools = AsyncMock(return_value=[
            {"name": "tool1", "description": "Tool 1", "input_schema": {}}
        ])
        server1.cleanup = AsyncMock()

        server2 = AsyncMock()
        server2.name = "sqlite"
        server2.list_tools = AsyncMock(return_value=[
            {"name": "read_query", "description": "Read query", "input_schema": {}},
            {"name": "write_query", "description": "Write query", "input_schema": {}},
        ])
        server2.cleanup = AsyncMock()

        return [server1, server2]

    @pytest.fixture
    def chat_session(self, mock_servers):
        """Create a ChatSession instance."""
        with patch("client.Anthropic") as mock_anthropic_class:
            mock_anthropic = Mock()
            mock_anthropic_class.return_value = mock_anthropic

            session = ChatSession(
                mock_servers,
                "test-api-key",
                api_url=None,
                model_name="test-model",
            )
            session.anthropic = mock_anthropic
            return session

    @pytest.mark.asyncio
    async def test_start_initializes_servers(self, chat_session, mock_servers):
        """Test that start() initializes all servers."""
        chat_session.chat_loop = AsyncMock()

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            try:
                await chat_session.start()
            except KeyboardInterrupt:
                pass

        for server in mock_servers:
            server.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_sets_sqlite_server(self, chat_session, mock_servers):
        """Test that start() identifies and sets the SQLite server."""
        chat_session.chat_loop = AsyncMock()

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            try:
                await chat_session.start()
            except KeyboardInterrupt:
                pass

        assert chat_session.sqlite_server is not None
        assert chat_session.sqlite_server.name == "sqlite"

    @pytest.mark.asyncio
    async def test_process_query_text_only(self, chat_session):
        """Test processing a query that returns only text."""
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = "This is a text response"

        mock_response = Mock()
        mock_response.content = [mock_content]

        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

        chat_session.anthropic.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_with_tool_use(self, chat_session, mock_servers):
        """Test processing a query that requires tool use."""
        # First response: tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "tool1"
        mock_tool_use.input = {"arg": "value"}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        # Second response: text only
        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Tool execution complete"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = [{"name": "tool1", "description": "Tool 1", "input_schema": {}}]
        chat_session.tool_to_server = {"tool1": "test_server"}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        # Mock the tool execution
        mock_tool_result = Mock()
        mock_tool_result.content = [Mock(text="Tool result")]
        mock_servers[0].execute_tool = AsyncMock(return_value=mock_tool_result)

        await chat_session.process_query("test query")

        # Should have called the API twice (initial + follow-up)
        assert chat_session.anthropic.messages.create.call_count == 2
        # Should have executed the tool
        mock_servers[0].execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_stored_data(self, chat_session):
        """Test showing stored data."""
        # Mock SQLite server
        mock_sqlite = AsyncMock()
        mock_text_content = Mock()
        mock_text_content.text = "[{'company_name': 'Test Co', 'plan_name': 'Test Plan', 'input_tokens': 0.1, 'output_tokens': 0.2, 'currency': 'USD'}]"

        mock_result = Mock()
        mock_result.content = [mock_text_content]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

        mock_sqlite.execute_tool.assert_called_once()
        call_args = mock_sqlite.execute_tool.call_args
        assert call_args[0][0] == "read_query"
        assert "SELECT company_name" in call_args[0][1]["query"]

    @pytest.mark.asyncio
    async def test_show_stored_data_no_sqlite(self, chat_session):
        """Test showing stored data when SQLite server is not available."""
        chat_session.sqlite_server = None

        await chat_session.show_stored_data()

        # Should not raise an error, just return early

    @pytest.mark.asyncio
    async def test_cleanup_servers(self, chat_session, mock_servers):
        """Test cleaning up all servers."""
        await chat_session.cleanup_servers()

        for server in mock_servers:
            server.cleanup.assert_called_once()

    def test_extract_url_from_result(self, chat_session):
        """Test URL extraction from result text."""
        result_text = "Visit https://example.com for more info"
        url = chat_session._extract_url_from_result(result_text)
        assert url == "https://example.com"

    def test_extract_url_from_result_no_url(self, chat_session):
        """Test URL extraction when no URL is present."""
        result_text = "No URL here"
        url = chat_session._extract_url_from_result(result_text)
        assert url is None

    @pytest.mark.asyncio
    async def test_process_query_api_error(self, chat_session):
        """Test process_query with API error."""
        chat_session.anthropic.messages.create.side_effect = Exception("API Error")
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

        # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_process_query_empty_response(self, chat_session):
        """Test process_query with empty response."""
        mock_response = Mock()
        mock_response.content = None

        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_tool_not_found(self, chat_session, mock_servers):
        """Test process_query when tool is not found."""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "nonexistent_tool"
        mock_tool_use.input = {}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Tool not found"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = []
        chat_session.tool_to_server = {}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_show_stored_data_parse_error(self, chat_session):
        """Test show_stored_data with unparseable data."""
        mock_sqlite = AsyncMock()
        mock_text_content = Mock()
        mock_text_content.text = "not valid python or json"

        mock_result = Mock()
        mock_result.content = [mock_text_content]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_exception(self, chat_session):
        """Test show_stored_data with exception."""
        mock_sqlite = AsyncMock()
        mock_sqlite.execute_tool = AsyncMock(side_effect=Exception("Database error"))

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_start_server_initialization_failure(self, chat_session, mock_servers):
        """Test start() when server initialization fails."""
        mock_servers[0].initialize = AsyncMock(side_effect=Exception("Init failed"))
        chat_session.chat_loop = AsyncMock()

        try:
            await chat_session.start()
        except Exception:
            pass

        # Should cleanup on failure
        mock_servers[0].cleanup.assert_called()

    @pytest.mark.asyncio
    async def test_process_query_no_assistant_content(self, chat_session):
        """Test process_query when no assistant content is generated."""
        mock_response = Mock()
        mock_response.content = []

        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_tool_result_dict_text(self, chat_session, mock_servers):
        """Test process_query with tool result as dict with text."""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "tool1"
        mock_tool_use.input = {}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Done"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = [{"name": "tool1", "description": "Tool 1", "input_schema": {}}]
        chat_session.tool_to_server = {"tool1": "test_server"}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        # Mock tool result as dict with text
        mock_tool_result = Mock()
        mock_tool_result.content = [{"text": "Tool result"}]
        mock_servers[0].execute_tool = AsyncMock(return_value=mock_tool_result)

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_tool_result_is_error(self, chat_session, mock_servers):
        """Test process_query with tool result that has isError."""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "tool1"
        mock_tool_use.input = {}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Error occurred"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = [{"name": "tool1", "description": "Tool 1", "input_schema": {}}]
        chat_session.tool_to_server = {"tool1": "test_server"}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        # Mock tool result with isError
        mock_tool_result = Mock()
        mock_tool_result.content = None
        mock_tool_result.isError = True
        mock_tool_result.text = None
        mock_servers[0].execute_tool = AsyncMock(return_value=mock_tool_result)

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_tool_result_no_content_no_text(self, chat_session, mock_servers):
        """Test process_query with tool result that has no content and no text."""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "tool1"
        mock_tool_use.input = {}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Done"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = [{"name": "tool1", "description": "Tool 1", "input_schema": {}}]
        chat_session.tool_to_server = {"tool1": "test_server"}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        # Mock tool result with no content and no text
        mock_tool_result = Mock()
        mock_tool_result.content = None
        del mock_tool_result.text  # Remove text attribute
        mock_tool_result.isError = False
        mock_servers[0].execute_tool = AsyncMock(return_value=mock_tool_result)

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_tool_result_non_list_content(self, chat_session, mock_servers):
        """Test process_query with tool result content that's not a list."""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "tool1"
        mock_tool_use.input = {}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Done"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = [{"name": "tool1", "description": "Tool 1", "input_schema": {}}]
        chat_session.tool_to_server = {"tool1": "test_server"}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        # Mock tool result with content that's not a list
        mock_tool_result = Mock()
        mock_tool_result.content = "string content"  # Not a list
        mock_servers[0].execute_tool = AsyncMock(return_value=mock_tool_result)

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_no_tool_use_blocks(self, chat_session):
        """Test process_query when response has text but no tool_use blocks."""
        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Just text response"

        mock_response = Mock()
        mock_response.content = [mock_text]

        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_process_query_empty_full_response(self, chat_session):
        """Test process_query when full_response is empty."""
        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "   "  # Only whitespace

        mock_response = Mock()
        mock_response.content = [mock_text]

        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

    @pytest.mark.asyncio
    async def test_show_stored_data_dict_text(self, chat_session):
        """Test show_stored_data with dict content containing text."""
        mock_sqlite = AsyncMock()
        mock_content_dict = {"text": '[{"company_name": "Test", "plan_name": "Plan", "input_tokens": 0.1, "output_tokens": 0.2, "currency": "USD"}]'}

        mock_result = Mock()
        mock_result.content = [mock_content_dict]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_dict_rows(self, chat_session):
        """Test show_stored_data with dict content containing rows."""
        mock_sqlite = AsyncMock()
        mock_content_dict = {"rows": [{"company_name": "Test", "plan_name": "Plan", "input_tokens": 0.1, "output_tokens": 0.2, "currency": "USD"}]}

        mock_result = Mock()
        mock_result.content = [mock_content_dict]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_dict_data(self, chat_session):
        """Test show_stored_data with dict content containing data."""
        mock_sqlite = AsyncMock()
        mock_content_dict = {"data": [{"company_name": "Test", "plan_name": "Plan", "input_tokens": 0.1, "output_tokens": 0.2, "currency": "USD"}]}

        mock_result = Mock()
        mock_result.content = [mock_content_dict]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_list_content(self, chat_session):
        """Test show_stored_data with list as content[0]."""
        mock_sqlite = AsyncMock()
        mock_content_list = [{"company_name": "Test", "plan_name": "Plan", "input_tokens": 0.1, "output_tokens": 0.2, "currency": "USD"}]

        mock_result = Mock()
        mock_result.content = [mock_content_list]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_non_dict_plan(self, chat_session):
        """Test show_stored_data with plan that's not a dict."""
        mock_sqlite = AsyncMock()
        mock_text_content = Mock()
        mock_text_content.text = "[{'company_name': 'Test'}, 'not a dict']"

        mock_result = Mock()
        mock_result.content = [mock_text_content]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_show_stored_data_text_data_is_list(self, chat_session):
        """Test show_stored_data when text_data is already a list."""
        mock_sqlite = AsyncMock()
        mock_text_content = Mock()
        mock_text_content.text = [{"company_name": "Test", "plan_name": "Plan", "input_tokens": 0.1, "output_tokens": 0.2, "currency": "USD"}]

        mock_result = Mock()
        mock_result.content = [mock_text_content]
        mock_sqlite.execute_tool = AsyncMock(return_value=mock_result)

        chat_session.sqlite_server = mock_sqlite

        await chat_session.show_stored_data()

    @pytest.mark.asyncio
    async def test_process_query_with_data_extractor(self, chat_session):
        """Test process_query when data_extractor is set."""
        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Response with pricing info"

        mock_response = Mock()
        mock_response.content = [mock_text]

        mock_data_extractor = AsyncMock()
        chat_session.data_extractor = mock_data_extractor
        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []

        await chat_session.process_query("test query")

        # Should call extract_and_store_data
        mock_data_extractor.extract_and_store_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_multiple_text_blocks(self, chat_session):
        """Test process_query with multiple text content blocks."""
        mock_text1 = Mock()
        mock_text1.type = "text"
        mock_text1.text = "First part"

        mock_text2 = Mock()
        mock_text2.type = "text"
        mock_text2.text = "Second part"

        mock_response = Mock()
        mock_response.content = [mock_text1, mock_text2]

        chat_session.anthropic.messages.create.return_value = mock_response
        chat_session.available_tools = []
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

    def test_chat_session_with_api_url(self):
        """Test ChatSession initialization with API URL."""
        with patch("client.Anthropic") as mock_anthropic_class:
            mock_anthropic = Mock()
            mock_anthropic_class.return_value = mock_anthropic

            session = ChatSession(
                [],
                "test-api-key",
                api_url="https://custom.url",
                model_name="test-model",
            )

            # Verify Anthropic was called with base_url
            mock_anthropic_class.assert_called_once_with(api_key="test-api-key", base_url="https://custom.url")
            assert session.anthropic == mock_anthropic
            assert session.model_name == "test-model"

    def test_chat_session_without_api_url(self):
        """Test ChatSession initialization without API URL."""
        with patch("client.Anthropic") as mock_anthropic_class:
            mock_anthropic = Mock()
            mock_anthropic_class.return_value = mock_anthropic

            session = ChatSession(
                [],
                "test-api-key",
                api_url=None,
                model_name="test-model",
            )

            # Verify Anthropic was called without base_url
            mock_anthropic_class.assert_called_once_with(api_key="test-api-key")
            assert session.anthropic == mock_anthropic
            assert session.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_process_query_server_not_found(self, chat_session, mock_servers):
        """Test process_query when server is not found."""
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "tool1"
        mock_tool_use.input = {}

        mock_response1 = Mock()
        mock_response1.content = [mock_tool_use]

        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Server not found"

        mock_response2 = Mock()
        mock_response2.content = [mock_text]

        chat_session.anthropic.messages.create.side_effect = [mock_response1, mock_response2]
        chat_session.available_tools = [{"name": "tool1", "description": "Tool 1", "input_schema": {}}]
        chat_session.tool_to_server = {"tool1": "nonexistent_server"}
        chat_session.servers = mock_servers
        chat_session.data_extractor = None

        await chat_session.process_query("test query")

