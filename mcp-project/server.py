import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from firecrawl import FirecrawlApp  # type: ignore

from mcp.server.fastmcp import FastMCP

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")


@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ["markdown", "html"],
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.

    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)

    Returns:
        List of provider names for successfully scraped websites
    """

    if api_key is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")

    app = FirecrawlApp(api_key=api_key)

    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)

    successful_scrapes: List[str] = []
    scraped_metadata: Dict[str, dict] = {}

    # Load existing metadata
    scraped_metadata_file = os.path.join(path, "scraped_metadata.json")

    try:
        with open(scraped_metadata_file, "r") as file:
            loaded_data = json.load(file)

            # Check if not empty
            if loaded_data:
                scraped_metadata = loaded_data

    except FileNotFoundError:
        logger.info("Scraped metadata file not found. Starting with empty metadata.")

    except json.JSONDecodeError:
        logger.error("Could not decode JSON from scraped metadata file.")

    except Exception as e:
        logger.error(f"Error reading scraped metadata file: {e}")

    # Loop through websites
    for provider_name, url in websites.items():
        try:
            logger.info(f"Scraping {provider_name}: {url}")
            scrape_result = app.scrape_url(url, formats=formats).model_dump()

            # Check if scrape was successful
            if not scrape_result.get("success", False):
                logger.error(f"Failed to scrape {provider_name}: {scrape_result.get('error', 'Unknown error')}")

                # Still add metadata entry for failed scrape
                scraped_metadata[provider_name] = {
                    "provider_name": provider_name,
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "scraped_at": datetime.now().isoformat(),
                    "formats": formats,
                    "success": False,
                    "content_files": {},
                    "title": "",
                    "description": "",
                }

                continue

            # Create metadata dictionary for this provider
            metadata = {
                "provider_name": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": True,
                "content_files": {},
                "title": scrape_result.get("title", ""),
                "description": scrape_result.get("description", ""),
            }

            # Save content for each format
            for format_type in formats:
                content = scrape_result.get(format_type, "")

                if content:
                    filename = f"{provider_name}_{format_type}.txt"
                    filepath = os.path.join(path, filename)

                    with open(filepath, "w", encoding="utf-8") as file:
                        file.write(content)

                    metadata["content_files"][format_type] = filename
                    logger.info(f"Saved {format_type} content to {filename}")

            # Add to successful scrapes
            successful_scrapes.append(provider_name)

            # Add metadata to main dictionary
            scraped_metadata[provider_name] = metadata

            logger.info(f"Successfully scraped {provider_name}: {url}")

        except Exception as e:
            logger.error(f"Error scraping website {url}: {e}")

            # Add failed entry to metadata
            scraped_metadata[provider_name] = {
                "provider_name": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": False,
                "content_files": {},
                "title": "",
                "description": "",
            }

    # Write entire scraped_metadata dictionary back to file
    with open(scraped_metadata_file, "w") as file:
        json.dump(scraped_metadata, file, indent=2)

    logger.info(f"Final results: Successfully scraped {len(successful_scrapes)} out of {len(websites)} websites")
    logger.info(f"Wrote scraped metadata to file: {scraped_metadata_file}")

    return successful_scrapes


@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.

    Args:
        identifier: The provider name, full URL, or domain to look for

    Returns:
        Formatted JSON string with the scraped information
    """

    logger.info(f"Extracting information for identifier: {identifier}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")

    # Load metadata
    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            scraped_metadata = json.load(file)

    except FileNotFoundError:
        return f"There's no saved information related to identifier '{identifier}'."

    except json.JSONDecodeError:
        return f"There's no saved information related to identifier '{identifier}'."

    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        return f"There's no saved information related to identifier '{identifier}'."

    # Find a match
    for provider_name, metadata in scraped_metadata.items():
        # Check if identifier matches provider_name, url, or domain
        if (
            identifier == provider_name
            or identifier == metadata.get("url", "")
            or identifier == metadata.get("domain", "")
        ):
            # Make a copy of the metadata
            result = metadata.copy()

            # Check if metadata has 'content_files'
            if "content_files" in result and result["content_files"]:
                # Create a new result['content'] dictionary
                result["content"] = {}

                # Loop through the content_files
                for format_type, filename in result["content_files"].items():
                    try:
                        # Read the content from the file
                        filepath = os.path.join(SCRAPE_DIR, filename)
                        with open(filepath, "r", encoding="utf-8") as file:
                            content = file.read()

                        result["content"][format_type] = content

                    except FileNotFoundError:
                        logger.warning(f"Content file not found: {filename}")
                        result["content"][format_type] = ""

                    except Exception as e:
                        logger.error(f"Error reading content file {filename}: {e}")
                        result["content"][format_type] = ""

            # Return the result dictionary as a formatted JSON string
            return json.dumps(result, indent=2)

    # If no match found
    return f"There's no saved information related to identifier '{identifier}'."


if __name__ == "__main__":
    print("Starting server...")
    mcp.run(transport="stdio")
