# MCP Project: LLM Inference Pricing Scraper

An MCP (Model Context Protocol) chatbot application that scrapes LLM inference serving websites, extracts pricing information, and stores it in a SQLite database. The system uses Anthropic Claude for natural language processing and tool orchestration.

## Features

- **Website Scraping**: Scrapes multiple websites using Firecrawl API and stores content in multiple formats (markdown, HTML)
- **Intelligent Data Extraction**: Uses Anthropic Claude to extract structured pricing data from scraped content
- **Data Storage**: Stores extracted pricing plans in SQLite database with company, plan details, token pricing, and features
- **Interactive Chatbot**: Natural language interface to query pricing information and compare providers
- **MCP Architecture**: Custom MCP server for scraping and integration with SQLite and filesystem MCP servers

## Setup

### Prerequisites

- Python 3.10+ (3.12 recommended)
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js (for filesystem MCP server)
- API keys:
  - Anthropic API key
  - Firecrawl API key

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. **Create `.env` file** in the project root:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ANTHROPIC_API_MODEL=claude-sonnet-4-5-20250929
   ANTHROPIC_API_URL=your_custom_url_if_needed  # Optional
   FIRECRAWL_API_KEY=your_firecrawl_api_key_here
   ```

4. **Verify installation:**
   ```bash
   uv run python -c "import anthropic, firecrawl, mcp; print('Dependencies OK')"
   ```

## Usage

### Running the Client Application

```bash
uv run python client.py
```

The application will:
1. Initialize MCP servers (scraping server, SQLite, filesystem)
2. Start an interactive chatbot session
3. Allow you to query pricing information using natural language

### Example Queries

- "Scrape pricing information from cloudrift.ai"
- "How much does cloudrift ai charge for deepseek v3?"
- "Compare cloudrift ai and deepinfra's costs for deepseek v3"
- "Show me the stored pricing data"

### Running the Scraper Server

The scraper server runs automatically when the client starts, but you can also run it standalone:

```bash
uv run python server.py
```

## Testing

### Automated Tests

Run the full test suite with coverage:

```bash
uv run pytest -v
```

This will:
- Run 86 tests covering both server and client functionality
- Generate coverage report (target: 85%+, current: 87.5%)
- Output HTML coverage report to `htmlcov/` directory

### Test Coverage

- **Server Tests**: Tests for `scrape_websites` and `extract_scraped_info` functions
- **Client Tests**: Tests for `Configuration`, `Server`, `DataExtractor`, and `ChatSession` classes
- **Coverage**: 87.5% overall coverage with comprehensive error handling tests

## Manual Testing

The following manual test scenarios can be used to verify end-to-end functionality:

### Initial Scraping

First, scrape these sites to populate the database:

```json
{
  "cloudrift": "https://www.cloudrift.ai/inference",
  "deepinfra": "https://deepinfra.com/pricing",
  "fireworks": "https://fireworks.ai/pricing#serverless-pricing",
  "groq": "https://groq.com/pricing"
}
```

### Test Queries

Use the following prompts in your chatbot to test different scenarios:

1. **Single Provider Query:**
   - "How much does cloudrift ai (https://www.cloudrift.ai/inference) charge for deepseek v3?"

2. **Alternative Provider Query:**
   - "How much does deepinfra (https://deepinfra.com/pricing) charge for deepseek v3"

3. **Comparison Query:**
   - "Compare cloudrift ai and deepinfra's costs for deepseek v3"

### Additional Testing

Feel free to play around with all the LLM providers in the list above. The system should:
- Successfully scrape websites and store content
- Extract pricing information using Claude
- Store data in SQLite database
- Retrieve and display stored data on request
- Handle errors gracefully

## Configuration

### Server Configuration (`server_config.json`)

The configuration file defines three MCP servers:

1. **llm_inference**: Custom scraping server (`server.py`)
2. **sqlite**: SQLite database server for data storage
3. **filesystem**: Filesystem access server

### Environment Variables

- `ANTHROPIC_API_KEY`: Required - Your Anthropic API key
- `ANTHROPIC_API_MODEL`: Required - Model name (e.g., `claude-3-5-sonnet-20240620`)
- `ANTHROPIC_API_URL`: Optional - Custom API URL (for proxies)
- `FIRECRAWL_API_KEY`: Required - Your Firecrawl API key

## Architecture

### MCP Server (`server.py`)

Provides two tools:
- **`scrape_websites`**: Scrapes websites using Firecrawl, saves content to files, and updates metadata
- **`extract_scraped_info`**: Retrieves scraped content and metadata for a given identifier

### MCP Client (`client.py`)

Orchestrates interactions between:
- Anthropic Claude (LLM for natural language processing)
- Custom scraping server
- SQLite MCP server (data storage)
- DataExtractor (structured data extraction and storage)

### Data Flow

1. User query → ChatSession
2. ChatSession → Anthropic Claude (with available tools)
3. Claude → Tool calls (scrape_websites, extract_scraped_info, SQL queries)
4. Tool results → Claude
5. Claude → Structured extraction (if pricing data found)
6. DataExtractor → SQLite database
7. Response → User

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

### Running Linters

```bash
uv run black .
uv run ruff check .
uv run mypy .
uv run isort .
```
