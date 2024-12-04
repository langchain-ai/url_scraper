"""URL Scraping using an LLM.

This is a simple example that shows how to scrape data from a URL
with an LLM.

It does not use any advanced features or try to set up an agent, and is meant
as a simple baseline that can be iterated upon.

The example comes with an evaluation set that can be used to evaluate
the performance of the LLM.
"""

from io import BytesIO
from typing import Any, Dict, Sequence, TypedDict

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from agent.configuration import Configuration
from agent.parsing import parse_binary_input
from agent.state import ExtractRequest, ExtractResponse, State
from agent.utils import load_chat_model, update_json_schema
from agent.validators import validate_json_schema

HEADERS = {
    # Set the user agent to identify the bot
    "User-Agent": (
        "MyAwesomeBot/1.0 (+https://mywebsite.com/bot-info; contact@mywebsite.com)"
    ),
}


async def fetch_data_from_url(url: str, proxies: str) -> bytes:
    """Fetch data from a URL."""
    client = httpx.AsyncClient(headers=HEADERS, proxies=proxies)
    response = await client.get(url)
    response.raise_for_status()
    return response.content


# PUBLIC API


async def extract_from_text(
    model_name: str, text: str, json_schema: Dict[str, Any]
) -> ExtractResponse:
    """Extract information from the given text.

    Args:
        model_name: The name of the model to use (e.g., "openai/gpt-4").
        text: The text to extract from (e.g., the text content of a web page).
        json_schema: The JSON schema to use for extraction.
            The extracted data will be a list of objects each of which
            must conform to this schema.

    Returns:
        ExtractResponse: The extracted data.
    """
    validate_json_schema(json_schema)
    schema = update_json_schema(json_schema)
    model = load_chat_model(model_name)
    structured_model = model.with_structured_output(
        schema=schema, method="function_calling"
    )

    system_message = {
        "content": (
            "You are a top-tier algorithm for extracting information from "
            "text. Only extract information that is relevant to the provided "
            "text. If no information is relevant, use the schema and output an "
            "empty list where appropriate."
        ),
        "role": "system",
    }

    human_message = {
        "content": (
            f"Please extract relevant information from the following "
            f"text:\n --- \n {text} \n ---"
        ),
        "role": "human",
    }

    prompt = [
        system_message,
        human_message,
    ]
    results = await structured_model.ainvoke(prompt)
    return ExtractResponse(data=results["data"])


async def extract(
    state: State,
    config: RunnableConfig,
) -> TypedDict(
    "ExtractedData",
    {
        "data": Sequence[dict],
    },
):
    """Fetch data from a URL and extract information using an LLM.

    Args:
        state (State): Contains a URL to scrape and a JSON schema.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        ExtractedData: The extracted data.
    """
    configuration = Configuration.from_runnable_config(config)
    binary_data = await fetch_data_from_url(state["url"], configuration.proxies)
    file = BytesIO(binary_data)
    file.name = state["url"]
    documents = parse_binary_input(file)

    if len(documents) > 1:
        raise ValueError("Expected a single document.")
    text = documents[0].page_content
    response = await extract_from_text(configuration.model, text, state["json_schema"])
    return {
        "data": response.data,
    }


# Define a new graph
graph_builder = StateGraph(
    State, config_schema=Configuration, input=ExtractRequest, output=ExtractResponse
)

# Add the node to the graph
graph_builder.add_node("extract", extract)

# Set entrypoint
graph_builder.add_edge("__start__", "extract")

# Compile the workflow into an executable graph
graph = graph_builder.compile()
graph.name = "URL Scraper"  # This defines the custom name in LangSmith
