import streamlit as st
import ast
from src.engine import create_engine, generate_form_completion
from data.examples import maintenance_requests
from llama_index.core import Settings
from models.models import llama3_8b_interface
import os

if __name__ == "__main__":

    # Add this check to ensure the script runs as a Streamlit app
    if os.getenv("_") is None or "streamlit" not in os.getenv("_"):
        os.system("streamlit run " + __file__)
    else:
        Settings.llm = llama3_8b_interface.llm
        query_engine, documents = create_engine(Settings.llm, maintenance_requests)

        # Streamlit application
        st.title("Interactive Form Completion")

        # Input for the summary
        summary = st.text_input("Enter a short summary of the issue:")

        if st.button("Generate Form"):

            # Generate the form using the RAG engine
            form = generate_form_completion(query_engine, summary)

            st.subheader("Generated Form")

            # Convert string representations of lists into actual lists
            department_list = ast.literal_eval(form["Department"])
            priority_list = ast.literal_eval(form["Priority"])
            description_list = ast.literal_eval(form["Description of Issue"])
            requested_acts_list = ast.literal_eval(form["Requested Actions"])
            formatted_requested_actions = "\n".join(
                f"{i+1}. {action}" for i, action in enumerate(requested_acts_list)
            )
            additional_notes_content = form["Additional Notes"].strip('"')

            # Display the generated form fields as editable inputs
            department = st.selectbox("Department", department_list, index=0)
            priority = st.selectbox("Priority", priority_list, index=0)
            description_of_issue = st.selectbox(
                "Description of Issue", description_list
            )
            requested_actions = st.text_area(
                "Requested Actions", formatted_requested_actions
            )
            additional_notes = st.text_area(
                "Additional Notes", additional_notes_content
            )
