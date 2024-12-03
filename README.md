# URL Scraper 🌐🤖

This repository is a starting point to test strategies for extracting structured information from a given URL using a language model.

## What's included

1. A bare minimum implementation that can be used to extract structured information from a given URL (`src/agent` folder).
2. A dataset and evaluation script to evaluate the performance of the agent doing the extraction (`src/eval` folder).

## How it works

The agent is a LangGraph agent that uses a language model to extract structured information from a given URL. The agent is implemented in the `src/agent` folder.

The agent does the following:

1. Accepts a URL and a JSON schema as input from a user.
2. Fetches the HTML content of a given URL.
3. Parses the HTML content into text.
4. Uses a vanilla chat model capable of [tool calling](https://python.langchain.com/docs/concepts/tool_calling/) to extract structured information from the text that matches the schema.

## 🚀 Launch the LangGraph Server

Install the langgraph CLI:

```shell
pip install "langgraph-cli[inmem]==0.1.61"
```

Install dependencies:

```shell
pip install -e .
```  

Load API keys into the environment for the LangSmith SDK and OpenAI API:

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

>> Ready!
>> 
>> API: http://127.0.0.1:2024
>> 
>> Docs: http://127.0.0.1:2024/docs
>> 
>> LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

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

A score between 0 and 1 is assigned to each extraction result by an LLM model that acts
as a judge.

The model assigns the score based on how closely the extracted information matches the expected information.

### Get the dataset

Create a new dataset in LangSmith using the code in the `eval` folder:

```shell
python eval/create_dataset.py
```

### Run the evaluation

To run the evaluation, you can use the `run_eval.py` script in the `eval` folder. This will create a new experiment in LangSmith for the dataset you created in the previous step.

```shell
python eval/run_eval.py --experiment-prefix "My custom prefix" --agent-url http://localhost:2024
```
