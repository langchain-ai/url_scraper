# URL Scraper 🌐🤖

This repository is a starting point to test strategies for extracting structured information from a specific URL using a language model.


## How it works

This repository implements the bare minimal implementation to get you started. It includes the following steps:

1. Fetches the HTML content of a given URL.
2. Parses the HTML content into a langchain `Document` object.
3. Extracts structured information from the `Document` using a user provided JSON schema and a chat model. Extraction relies on vanilla [tool calling](https://python.langchain.com/docs/concepts/tool_calling/)

## 🚀 Launch the LangGraph Server

Install the langgraph CLI:

```shell
pip install "langgraph-cli[inmem]==0.1.61"
```

Install dependencies:

```shell
pip install -e .
```  

Load API Keys into the environment for the LangSmith SDK and OpenAI API:

```shell
export LANGSMITH_API_KEY=<your_langsmith_api_key>
# Or configure another chat model
export OPENAI_API_KEY=<your_openai_api_key>
```

Launch the agent:

```shell
langgraph dev
```

If all is well, you should see the following output:

```shell
Ready!

API: http://127.0.0.1:2024

Docs: http://127.0.0.1:2024/docs

LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Improving the agent

You can try to improve the extraction strategy in a variety of ways. For example,

1. Improving the HTML parsing strategy.
2. Adding handling of large html documents and deduplication of extracted information.
3. Adding reflection steps.
4. Extend this to work with data URLs and accept other file formats like PDFs. (The `src/agent/parsing` already has functionality to parse PDFs, you just need to hook it up.)

## Evaluation

Prior to engaging in any optimization, it is important to establish a baseline performance. This repository includes:

1. A dataset consisting of a list of URLs and the expected structured information to be extracted from each URL.
2. An evaluation script that can be used to evaluate the agent on this dataset.

### Set up

Make sure you have the LangSmith CLI installed:

```shell
pip install langsmith-cli
```

And set your API key:

```shell
export LANGSMITH_API_KEY=<your_langsmith_api_key>
```

### Evaluation metric

The quality of the extraction results is evaluated by an LLM model which assigns a
score between 0 and 1 depending on how closely the extracted information matches the expected information.

### Get the dataset

Make sure that you have the LangSmith CLI installed:

Create a new dataset in LangSmith using the code in the `eval` folder:

```shell
python eval/create_dataset.py
```

### Run the evaluation

To run the evaluation, you can use the `run_eval.py` script in the `eval` folder. This will create a new experiment in LangSmith for the dataset you created in the previous step.

```shell
python eval/run_eval.py --experiment-prefix "My custom prefix" --agent-url http://localhost:2024
```
