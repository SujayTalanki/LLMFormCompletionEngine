"""
This file tests the LLM's performance
"""

from llama_index.core import Settings
from data.examples import maintenance_requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from engine import create_engine, generate_form_completion
from rouge_score import rouge_scorer
from bert_score import score
import json
from src.prompts import generate_data, generating_augmented_summaries_prompt
from collections import Counter


def calculate_rouge(generated, reference):
    """
    Calculates the Rouge score based on the generated description
    and input description

    Args:
        generated (str): The llm generated output.
        reference (str): The actual description.

    Returns:
        float: Rouge score metric for the input description
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )  # noqa
    scores = scorer.score(reference, generated)
    return scores


def calculate_bert(generated, reference):
    """
    Calculates the BERT similarity score based on the
    generated description and actual description.

    Args:
        generated (str): The llm generated output.
        reference (str): The actual description.

    Returns:
        float: BERT score metric for the input description
    """
    P, R, F1 = score([generated], [reference], lang="en", verbose=True)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def write_results_to_file(
    filename,
    department_results,
    priority_results,
    generated_summaries,
    generated_descriptions,
    expected_descriptions,
):
    """
    Writes the LLM results to the designated filepath.

    Args:
        filename (str): Output filename.
        department_results (str): LLM prediction of department.
        priority_results (str): LLM prediction of priority.
        generated_summaries (str): LLM prediction of summary.
        generated_descriptions (str): LLM prediction of description.
        expected_descriptions (str): Actual description for reference.

    Returns:
        None.
    """
    # Formats the results for easy reasability.
    with open(filename, "w") as file:
        file.write("Department Scores:\n")
        file.write("-----------------------------------------\n")
        file.write(f"{department_results}\n\n")

        file.write("\n")
        file.write("-----------------------------------------\n")
        file.write("\n")

        file.write("Priority Scores:\n")
        file.write("-----------------------------------------\n")
        file.write(f"{priority_results}\n\n")

        for generated, expected, summary in zip(
            generated_descriptions, expected_descriptions, generated_summaries
        ):
            file.write(f"Generated Summary: {summary}\n")
            file.write("\n")
            file.write(f"Generated Description: {generated}\n")
            file.write("\n")
            file.write(f"Actual Description: {expected}\n")
            file.write("\n")
            bert = calculate_bert(generated, expected)
            file.write(f"BERT Score: {bert}\n")
            file.write("\n")
            file.write("-----------------------------------------\n")
            file.write("\n")


def write_results_to_json(results):
    """
    Writes the results to json format.

    Args:
        results (list): list of completed forms.

    Returns:
        None. The resulting file will have the completed
              forms in json format for future downstream
              tasks.
    """
    with open("model_results/forms.json", "w") as file:
        json.dump(results, file, indent=4)


def generate_rephrased_summary(description):
    """
    Uses gpt-4o to generate synthetic summaries of
    potential problems for model testing. This function
    takes in a short description of the problem, and gpt-4o
    generates many summaries for a variety of examples.

    Args:
        description (str): The description of a problem.

    Returns:
        str: a rephrased summary of the description.
    """
    rephrased_summary = generate_data(
        system_prompt=generating_augmented_summaries_prompt,
        description=description,
        augmented=True,
        max_tokens=500,
        temperature=0.9,
    )
    return rephrased_summary


def most_common_responses(generated_dictionaries):
    """
    Extracts the most common department and priority from the simulated
    responses.

    Args:
        generated_dictionaries (list): List containing the completed forms
                                       in dictionary format.

    Returns:
        list: The most probable priority and department, as well as the
              a generated description.
    """
    # Generated priorities and departments
    priorities = [item["Priority"] for item in generated_dictionaries]
    departments = [item["Department"] for item in generated_dictionaries]

    # Extracts most common priority and description
    most_common_priority = Counter(priorities).most_common(1)[0]
    most_common_department = Counter(departments).most_common(1)[0]

    return [
        most_common_priority[0],
        most_common_department[0],
        generated_dictionaries[0]["Description of Issue"],
    ]


def evaluate_performance(query_engine, documents, filename):
    """
    Combines the functions to create a testing pipeline.

    Args:
        query_engine (QueryEngine): Query engine for RAG system.
        documents (list): list of document objects for RAG system.
        filename (str): Output filename.

    Returns:
        None: The function will write the results to the designated files.
    """
    # Actual form fields
    expected_departments = [doc.metadata["Department"] for doc in documents]
    expected_priorities = [doc.metadata["Priority"] for doc in documents]
    expected_descriptions = [
        doc.metadata["Description of Issue"] for doc in documents
    ]  # noqa

    # Lists for LLM predictions
    generated_forms = []
    generated_departments = []
    generated_priorities = []
    generated_descriptions = []
    generated_summaries = []

    # Iterates through LLM output to parse the fields for formatting
    for description in expected_descriptions:
        rephrased_summary = generate_rephrased_summary(description)
        completed_form = generate_form_completion(
            query_engine, rephrased_summary
        )  # noqa
        generated_priorities.append(completed_form["Priority"])
        generated_departments.append(completed_form["Department"])
        generated_descriptions.append(completed_form["Description of Issue"])
        generated_summaries.append(rephrased_summary)
        generated_forms.append(completed_form)

    # Sklearn function to calculate metrics
    department_results = classification_report(
        expected_departments, generated_departments
    )
    priority_results = classification_report(
        expected_priorities, generated_priorities  # noqa
    )

    # Code to write the results in a readable format + json format for future.
    write_results_to_file(
        filename,
        department_results,
        priority_results,
        generated_summaries,
        generated_descriptions,
        expected_descriptions,
    )
    write_results_to_json(generated_forms)


if __name__ == "__main__":

    # Model for embedding transformation
    model = SentenceTransformer("all-MiniLM-L6-v2")
    filename = "model_results/mixtral_7b_results.txt"

    # Engine that preprocesses, runs the LLM, and generated the documents
    query_engine, documents = create_engine(Settings.llm, maintenance_requests)
    evaluate_performance(query_engine, documents, filename)
