import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from anthropic import Anthropic # type: ignore
from dotenv import load_dotenv # type: ignore
from mcp import ClientSession, StdioServerParameters # type: ignore
from mcp.client.stdio import stdio_client # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = os.getenv("ANTHROPIC_API_MODEL")
        # Get API URL, but treat empty strings as None (for commented out env vars)
        api_url = os.getenv("ANTHROPIC_API_URL")
        self.api_url = api_url if api_url and api_url.strip() else None

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        try:
            with open(file_path, "r") as f:
                config = json.load(f)

            if "mcpServers" not in config:
                raise ValueError("Configuration file is missing required 'mcpServers' field")

            return config

        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}") from e

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        return self.api_key

    @property
    def anthropic_model(self) -> str:
        """Get the Anthropic model name.

        Returns:
            The model name as a string.
        """
        return self.model_name

    @property
    def anthropic_api_url(self) -> str | None:
        """Get the Anthropic API URL (if custom proxy is used).

        Returns:
            The API URL as a string, or None if using default.
        """
        return self.api_url


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError("Server session not initialized. Call initialize() first.")

        tools_response = await self.session.list_tools()
        tools_list: List[ToolDefinition] = []

        for tool in tools_response.tools:
            tool_def: ToolDefinition = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            tools_list.append(tool_def)

        return tools_list

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError("Server session not initialized. Call initialize() first.")

        last_exception: Exception | None = None

        for attempt in range(retries + 1):
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(
                    name=tool_name, arguments=arguments, read_timeout_seconds=timedelta(seconds=60)
                )
                return result

            except Exception as e:
                last_exception = e

                if attempt < retries:
                    logging.warning(f"Tool execution failed (attempt {attempt + 1}/{retries + 1}): {e}. Retrying...")
                    await asyncio.sleep(delay)

                else:
                    logging.error(f"Tool execution failed after {retries + 1} attempts: {e}")

        if last_exception:
            raise last_exception

        raise RuntimeError("Tool execution failed with unknown error")

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""

    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic, model_name: str):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client
        self.model_name = model_name

    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:

            await self.sqlite_server.execute_tool(
                "write_query",
                {
                    "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                    features TEXT,  -- JSON array
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
                },
            )

            logging.info("✓ Data extraction tables initialized")

        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            response = self.anthropic.messages.create(
                max_tokens=1024, model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )

            text_content = ""
            for content in response.content:
                if content.type == "text":
                    text_content += content.text

            return text_content.strip()

        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'

    async def extract_and_store_data(self, user_query: str, llm_response: str, source_url: str | None = None) -> None:
        """Extract structured data from LLM response and store it."""
        try:
            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format:
            
            Text: {llm_response}
            
            Extract pricing plans with this structure:
            {{
                "company_name": "company name",
                "plans": [
                    {{
                        "plan_name": "plan name",
                        "input_tokens": number or null,
                        "output_tokens": number or null,
                        "currency": "USD",
                        "billing_period": "monthly/yearly/one-time",
                        "features": ["feature1", "feature2"],
                        "limitations": "any limitations mentioned",
                        "query": "the user's query"
                    }}
                ]
            }}
            
            Return only valid JSON, no other text. Do not return your response enclosed in ```json```
            """

            extraction_response = await self._get_structured_extraction(extraction_prompt)
            extraction_response = extraction_response.replace("```json\n", "").replace("```", "")
            pricing_data = json.loads(extraction_response)

            for plan in pricing_data.get("plans", []):
                await self.sqlite_server.execute_tool(
                    "write_query",
                    {
                        "query": f"""
                    INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
                    VALUES (
                        '{pricing_data.get("company_name", "Unknown")}',
                        '{plan.get("plan_name", "Unknown Plan")}',
                        '{plan.get("input_tokens", 0)}',
                        '{plan.get("output_tokens", 0)}',
                        '{plan.get("currency", "USD")}',
                        '{plan.get("billing_period", "unknown")}',
                        '{json.dumps(plan.get("features", []))}',
                        '{plan.get("limitations", "")}',
                        '{user_query}')
                    """
                    },
                )

            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans")

        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(
        self,
        servers: list[Server],
        api_key: str,
        api_url: str | None = None,
        model_name: str = "claude-3-5-sonnet-20240620",
    ) -> None:
        self.servers: list[Server] = servers
        self.model_name = model_name
        # Initialize Anthropic client with custom URL if provided, else use default
        if api_url:
            logging.info(f"Using custom Anthropic API URL: {api_url}")
            self.anthropic = Anthropic(api_key=api_key, base_url=api_url)
        else:
            logging.info("Using default Anthropic API URL")
            self.anthropic = Anthropic(api_key=api_key)
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in self.available_tools:
            anthropic_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
            )

        messages: List[Any] = [{"role": "user", "content": query}]

        try:
            logging.info(f"Calling Anthropic API with model: {self.model_name}")
            response = self.anthropic.messages.create(
                max_tokens=2024, model=self.model_name, tools=anthropic_tools, messages=messages  # type: ignore[arg-type]
            )
            logging.info(
                f"Received response, content type: {type(response.content)}, length: {len(response.content) if response.content else 0}"
            )
        except Exception as e:
            logging.error(f"Error calling Anthropic API: {e}")
            print(f"Error: Failed to get response from API: {str(e)}")
            return

        full_response = ""
        source_url: str | None = None
        used_web_search = False

        process_query = True
        while process_query:
            assistant_content: List[Any] = []
            # Check if response.content exists and is iterable
            if response.content is None:
                logging.error("Response content is None")
                print("Error: Received empty response from API")
                break
            if not hasattr(response, "content") or not response.content:
                logging.error("Response has no content or content is empty")
                print("Error: Response has no content")
                break
            for content in response.content:
                if content.type == "text":
                    full_response += content.text + "\n"
                    assistant_content.append({"type": "text", "text": content.text})
                    # Check if this is the only content
                    if len(response.content) == 1:
                        process_query = False
                elif content.type == "tool_use":
                    # Append the tool use request to assistant_content and then to messages
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": content.id,
                            "name": content.name,
                            "input": content.input,
                        }
                    )
                    messages.append({"role": "assistant", "content": assistant_content})

                    # Get the tool id, args, and name from the content
                    tool_id = content.id
                    tool_name = content.name
                    tool_args = content.input

                    # Find the server that has this tool
                    server_name = self.tool_to_server.get(tool_name)
                    if not server_name:
                        tool_result_content = f"Error: Tool '{tool_name}' not found"
                    else:
                        # Find the server object
                        server = next((s for s in self.servers if s.name == server_name), None)
                        if not server:
                            tool_result_content = f"Error: Server '{server_name}' not found"
                        else:
                            # Execute tool on that server
                            try:
                                result = await server.execute_tool(tool_name, tool_args)
                                # MCP call_tool returns a CallToolResult with content attribute
                                if hasattr(result, "content") and result.content is not None:
                                    # Extract text from content items
                                    text_parts = []
                                    # Ensure content is iterable
                                    if isinstance(result.content, (list, tuple)):
                                        for item in result.content:
                                            if hasattr(item, "text"):
                                                text_parts.append(item.text)
                                            elif isinstance(item, dict) and "text" in item:
                                                text_parts.append(item["text"])
                                            else:
                                                text_parts.append(str(item))
                                    else:
                                        # If content is not a list, try to convert it
                                        text_parts.append(str(result.content))
                                    tool_result_content = "\n".join(text_parts) if text_parts else "No content returned"
                                else:
                                    # If no content attribute or content is None, try to get text from result
                                    if hasattr(result, "text"):
                                        tool_result_content = result.text
                                    elif hasattr(result, "isError") and result.isError:
                                        tool_result_content = f"Tool execution error: {str(result)}"
                                    else:
                                        tool_result_content = str(result) if result else "No result returned"
                            except Exception as e:
                                tool_result_content = f"Error executing tool: {str(e)}"

                    # Append the tool_result to messages
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool-result",
                                    "tool_use_id": tool_id,
                                    "content": tool_result_content,
                                }
                            ],
                        }
                    )

                    # Call self.anthropic.messages.create again with the new messages list
                    try:
                        response = self.anthropic.messages.create(
                            max_tokens=2024, model=self.model_name, tools=anthropic_tools, messages=messages  # type: ignore[arg-type]
                        )
                        logging.info(
                            f"Received follow-up response, content length: {len(response.content) if response.content else 0}"
                        )
                    except Exception as e:
                        logging.error(f"Error calling Anthropic API in tool loop: {e}")
                        print(f"Error: Failed to get follow-up response: {str(e)}")
                        process_query = False
                        break

                    # Check if response is valid
                    if response.content is None or not response.content:
                        logging.error("Follow-up response has no content")
                        print("Error: Follow-up response has no content")
                        process_query = False
                        break

                    # Check if the new response is just text, and if so, stop the loop
                    if len(response.content) == 1 and response.content[0].type == "text":
                        full_response += response.content[0].text + "\n"
                        process_query = False

            # If we processed text content and it was the only content, we already set process_query = False
            if assistant_content and process_query:
                messages.append({"role": "assistant", "content": assistant_content})

        # Print the final response
        if full_response.strip():
            print(full_response.strip())

        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(query, full_response.strip(), source_url or "")

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break
                elif query.lower() == "show data":
                    await self.show_stored_data()
                    continue

                await self.process_query(query)
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return

        try:
            pricing = await self.sqlite_server.execute_tool(
                "read_query",
                {
                    "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC LIMIT 5"
                },
            )

            print("\nRecently Stored Data:")
            print("=" * 50)

            print("\nPricing Plans:")
            # The result.content is a list with one item, a dict, where the 'text' key holds the rows
            if pricing.content and len(pricing.content) > 0 and "text" in pricing.content[0]:
                for plan in pricing.content[0]["text"]:
                    print(
                        f"  • {plan['company_name']}: {plan['plan_name']} - Input Token ${plan['input_tokens']}, Output Tokens ${plan['output_tokens']}"
                    )
            else:
                print("  No pricing plans found.")

            print("=" * 50)
        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")

            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic, self.model_name)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()

    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"

    server_config = config.load_config(config_file)

    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(
        servers,
        config.anthropic_api_key,
        api_url=config.anthropic_api_url,
        model_name=config.anthropic_model,
    )
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
