from math import isclose
from typing import Optional

from langchain_openai import ChatOpenAI
from langgraph.pregel.remote import RemoteGraph
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResults
from pydantic import BaseModel, Field

client = Client()

DEFAULT_DATASET_NAME = "Webpage Extraction Dataset"
DEFAULT_GRAPH_ID = "agent"
DEFAULT_AGENT_URL = "https://langr.ph/marketplace/b5152a8c-e0ed-4ed9-9bac-9c3cb7566c8d"

JUDGE_LLM = ChatOpenAI(model="gpt-4o")


def evaluate_agent_llm_judge(outputs: dict, reference_outputs: dict):
    """Evaluate the model's output against the expected output."""
    actual = outputs.get("data", [])
    expected = reference_outputs["output"]

    class Score(BaseModel):
        """Evaluate the model's output against the expected output."""

        score: float = Field(
            description="A score between 0 and 1 indicating the accuracy of the "
            "model's output compared to the expected output. 1 is a "
            "perfect match."
        )
        reason: str = Field(
            description="A brief explanation for why you scored the model's output as "
            "you did."
        )

    score = JUDGE_LLM.with_structured_output(Score).invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert at extracting structured data based on a schema. "
                    "Your task is to evaluate the accuracy of the model's output based on the expected output. "
                    "Penalize heavily instances when the model extracts results, but no results should be extracted.\n"
                    "Use the following guidelines to assign a score:\n "
                    " * Score = 0.0: No output is expected, but the model extracts content, or the output is entirely incorrect or irrelevant.\n"
                    " * Score = 0.25: The output shows some resemblance to the expected result but largely misinterprets or misrepresents the task.\n"
                    " * Score = 0.5: The output captures some correct elements but omits or misrepresents significant parts.\n"
                    " * Score = 0.75: The output is mostly correct with minor errors or omissions that don't significantly impact accuracy.\n"
                    " * Score = 1.0: The output matches the expected result perfectly with no errors or omissions.\n"
                ),
            },
            {
                "role": "user",
                "content": f"Actual output: {actual}\nExpected output: {expected}",
            },
        ]
    )
    return score.score


def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> int:
    """Evaluator function to check if the extracted answer is correct.

    This function compares the extracted answer with the reference answer.

    If both are numbers, it checks if they are close enough and returns 1 if they are.

    It penalizes wrong answers with -1.

    If the provided answer is None (was not answered), then the score wil be 0
    """
    actual_answer = outputs["value"]
    reference_answer = reference_outputs["answer"]

    if reference_answer is None:
        return 1 if actual_answer is None else -1

    if actual_answer is None:
        # We assign a score of 0 if the answer is not provided
        return 0

    try:
        actual_float = float(actual_answer)
        reference_float = float(reference_answer)
    except (ValueError, TypeError):
        # Return False if conversion to float fails
        return 0

    # Compare the floats with precision tolerance
    is_correct = isclose(actual_float, reference_float, rel_tol=1e-9, abs_tol=1e-9)
    return 1 if is_correct else -1


def make_agent_runner(graph_id: str, agent_url: str):
    agent_graph = RemoteGraph(graph_id, url=agent_url)

    def run_agent(inputs: dict):
        try:
            result = agent_graph.invoke(
                {"url": inputs["url"], "json_schema": inputs["json_schema"]}
            )

            return {"data": result["data"]}
        except:
            return {
                "data": [],
            }

    return run_agent


def get_agent_metadata(graph_id: str, agent_url: str):
    if "marketplace" in agent_url:
        project_id = agent_url.split("/")[-1]
        return {"project_id": project_id, "graph_id": graph_id}
    return {"graph_id": graph_id}


def run_eval(
    *,
    dataset_name: str,
    graph_id: str = DEFAULT_GRAPH_ID,
    agent_url: str = DEFAULT_AGENT_URL,
    experiment_prefix: Optional[str] = None,
    min_score: Optional[float] = None,
) -> EvaluationResults:
    dataset = client.read_dataset(dataset_name=dataset_name)
    run_agent = make_agent_runner(graph_id, agent_url)
    results = evaluate(
        run_agent,
        data=dataset,
        evaluators=[evaluate_agent_llm_judge],
        experiment_prefix=experiment_prefix,
        metadata=get_agent_metadata(graph_id, agent_url),
    )

    if min_score is not None:
        results_df = results.to_pandas()
        score = results_df["feedback.correct"].mean()
        if score < min_score:
            raise AssertionError(
                f"Average fraction of correctly extracted fields ({score}) "
                f"is less than min expected score of {min_score}"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name of the dataset to evaluate against",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=DEFAULT_GRAPH_ID,
        help="ID of the graph to evaluate",
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default=DEFAULT_AGENT_URL,
        help="URL of the deployed agent to evaluate",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        help="Experiment prefix for the evaluation",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum acceptable score for evaluation",
    )
    args = parser.parse_args()

    run_eval(
        dataset_name=args.dataset_name,
        graph_id=args.graph_id,
        agent_url=args.agent_url,
        experiment_prefix=args.experiment_prefix,
        min_score=args.min_score,
    )
