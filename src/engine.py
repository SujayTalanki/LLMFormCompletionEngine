"""
This file creates the pipeline for form completion
"""

import replicate
import ast
from data.examples import maintenance_requests
from src.prompts import (
    form_completion_prompt,
    interface_form_completion_prompt,
    feedback_prompt,
)
from llama_index.core import VectorStoreIndex, Document, Settings
from models.models import (
    llama3_8b,
    mixtral_7b,
    llama3_8b_interface,
    llama3_8b_feedback,
)


def direct_parse_response(response_text):
    """
    Converts the string output into a dictionary for future function calls

    Args:
        response_text (str): String containing the document fields and
                                descriptions.

    Returns:
        dict: Dictionary structure that contains the fields as the keys
                    and the descriptions as values.
    """

    # Split the response by lines and parse each line into a dictionary entry
    lines = response_text.split("\n")
    response_dict = {}

    # Iterates through lines and formats fields as keys, descriptions as values
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip()
            value = value.strip()
            value = ast.literal_eval(value)
            response_dict[key] = value

    return response_dict


def create_document(entry):
    """
    Creates document objects from the RAG documents in dictionary form

    Args:
        entry (dict): Dictionary that respresents the contents of the
                            document.

    Returns:
        Document: A Document object containing the form information.
    """

    document = Document(
        text="",
        metadata={
            "Form Type": entry["Form Type"],
            "Request ID": entry["Request ID"],
            "Date": entry["Date"],
            "Requested By": entry["Requested By"],
            "Department": entry["Department"],
            "Priority": entry["Priority"],
            "Description of Issue": entry["Description of Issue"],
            "Additional Notes": entry["Additional Notes"],
        },
    )

    return document


def create_engine(llm_config, requests):
    """
    Creates query engine for the RAG system based on the passed in
    LLM configuration and documents (requests)

    Args:
        llm_config (Replicate): Replicate (API) object that represents
                                a large language model (LLM).
        requests (list): A list of maintanence request forms (represented as
                         as dictionaries).

    Returns:
        tuple: Tuple containing the query engine and documents for future
               use.
    """

    # Creates list of document objects
    documents = [create_document(entry) for entry in requests]

    # Creates index for document storages
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm_config,
    )

    # Retriever to fetch the top 5 most similar documents
    retriever = index.as_retriever(search_type="similarity", top_k=5)

    # creates the query engine and returns the document object for future calls
    query_engine = index.as_query_engine(retriever=retriever, llm=llm_config)
    return query_engine, documents


# Pipeline that generates the form completion and outputs info to dictionary
def generate_form_completion(engine, description):
    """
    Pipeline that generates the form completion and outputs the information
    into a dictionary.

    Args:
        engine (QueryEngine): RAG based query engine
        description (str): A description of the problem that will be used to
                           autocomplete the rest of the form.

    Returns:
        dict: A dictionary containing the completed form fields.
    """

    # Constructs the prompt using the description
    prompt = (
        f"Now, please generate the corresponding form completion fields given "
        f"the description, according to prompt details stated and the format "
        f"below:\nSummary of problem: '{description}'\n"
        f"Fill out the fields in the following format:\n"
        f"Department: <Department>\n"
        f"Priority: <Priority>\n"
        f"Description of Issue: <Description of Issue>\n"
        f"Requested Actions: <Requested Actions>\n"
        f"Additional Notes: <Additional Notes>\n"
    )

    # Inputs prompt, system prompt, and description into form completion engine
    response = engine.query(interface_form_completion_prompt + "\n" + prompt)
    response_text = response.response

    # Directly parse the response into a dictionary
    response_dict = direct_parse_response(response_text)
    return response_dict


def generate_feedback(input, model_path):
    """
    Takes in initial form completion and makes adjustments according to
    user feedback.

    Args:
        input (str): User description of what needs to be changed to
                     improve form completion.

    Returns:
        dict: A dictionary containing the regenerated fields according
              to feedback.
    """

    # Constructs the prompt using user input
    prompt = (
        f"Given the following dictionary, please fill out the following fields"
        f" below in accordance to the system prompt. Please make sure to"
        f" update the fields in the keys of the 'Fields to Regenerate'"
        f" dictionary with the details included in the respective values.\n"
        f"Input dictionary: {input}\n"  # {input} is a dynamic variable
        f"Fill out the fields in the following format. ONLY include this"
        f" information, do NOT include any information about text"
        f" generation:\nDepartment: <Department>\n"
        f"Priority: <Priority>\n"
        f"Description of Issue: <Description of Issue>\n"
        f"Requested Actions: <Requested Actions>\n"
        f"Additional Notes: <Additional Notes>\n"
    )

    # System to regenerate form based on user feedback
    output = ""
    for event in replicate.stream(
        model_path,
        input={
            "prompt": prompt,
            "temperature": 0.7,
            "system_prompt": feedback_prompt,
            "length_penalty": 1,
            "max_new_tokens": 2500,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "prompt_template": (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "{system_prompt}"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                "{prompt}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "presence_penalty": 0,
        },
    ):
        output += str(event)

    # Parses llm output into dictionary format
    response_dict = direct_parse_response(output)
    return response_dict


if __name__ == "__main__":

    # Setting up RAG system
    Settings.llm = llama3_8b_interface.llm
    query_engine, documents = create_engine(Settings.llm, maintenance_requests)

    print(
        generate_form_completion(
            query_engine,
            "There have been HVAC system leaks",
        )
    )

    # Autocompleted form that as an input to feedback model (not tested!)
    initial_form = {
        "Department": ["Engineering", "Mechanical", "Electrical"],
        "Priority": ["Medium", "Low", "High"],
        "Description of Issue": [
            "There have been leaks detected in the HVAC system, indicating a "
            "potential issue with the system's integrity. A thorough "
            "inspection is required to identify the source and extent of  "
            "the leaks, and to determine the necessary repairs or maintenance "
            "to ensure the system operates safely and efficiently.",
            "Leaks have been reported in the HVAC system, necessitating an "
            "immediate investigation to identify the cause and scope of the "
            "issue. The system's performance and safety may be compromised, "
            "and prompt action is required to prevent further damage or "
            "potential hazards.",
            "The presence of leaks in the HVAC system suggests a malfunction "
            "or  defect, which demands a prompt and thorough examination to "
            "determine  the root cause and necessary corrective actions. "
            "The system's reliability and safety are at risk, and swift "
            "attention is required to prevent any further issues or "
            "disruptions.",
        ],
        "Requested Actions": [
            "Inspect the HVAC system for leaks",
            "Identify and repair or replace damaged components",
            "Verify system performance post-repair",
        ],
        "Additional Notes": (
            "Previous maintenance records should be reviewed to determine if "
            "any recent work may have contributed to the leaks. Crew reports "
            "of unusual noises or odors should also be investigated."
        ),
        "Fields to Regenerate": {
            "Description of Issue": (
                "Please include details about a broken radiator and faulty "
                "valve."  # noqa
            )
        },
    }

    # System prompt for this hasn't been completed, comment out
    # print(generate_feedback(initial_form))
