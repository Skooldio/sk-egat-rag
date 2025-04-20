# Import necessary libraries
import os  # For interacting with the operating system (e.g., environment variables)
import streamlit as st  # For accessing Streamlit secrets
from langchain.chat_models import init_chat_model  # Function to initialize chat models from LangChain
from google.oauth2 import service_account  # For handling Google Cloud service account credentials
from langchain_core.prompts import PromptTemplate # For creating prompt templates

# Set the OpenAI API key from Streamlit secrets.
# This allows using OpenAI models if selected.
os.environ["OPENAI_API_KEY"] = st.secrets.llm.openai_api_key

# Load Google Cloud service account credentials from a local JSON file.
# Required for using Google Vertex AI models.
egat_service_account = service_account.Credentials.from_service_account_file(filename="./service_account.json")

# Initialize the desired chat model.
# Option 1 (Commented out): Initialize OpenAI's gpt-4o-mini model.
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# Option 2 (Active): Initialize Google's Gemini 1.0 Pro model via Vertex AI, using the loaded credentials.
llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai", credentials=egat_service_account)

# Define a prompt template string.
# This template instructs the LLM to respond sarcastically to the user's prompt.
# The `{user_prompt}` placeholder will be filled with the actual user input.
template="""
try to answer with sarcastic based on the following prompt:

Prompt:{user_prompt}
"""

# Define the main function to generate LLM responses.
def llm_response(messages):
    """
    Takes a list of messages, extracts the latest user prompt, formats it
    using the defined template, sends it to the initialized LLM, and returns
    the LLM's response content.

    Args:
        messages (list): A list of message dictionaries, where the last element
                         is assumed to be the user's latest message.

    Returns:
        str: The content of the language model's response.
    """
    # Extract the content of the last message in the list (the user's prompt).
    user_prompt = messages[-1]["content"]
    # Print the user prompt to the console for debugging purposes.
    print ("user_prompt", user_prompt)

    # Create a PromptTemplate object from the template string.
    custom_template = PromptTemplate.from_template(template)

    # Format the prompt by filling in the 'user_prompt' placeholder.
    prompt = custom_template.invoke({"user_prompt": user_prompt})

    # Print the final formatted prompt to the console for debugging.
    print ("Final Prompt", prompt)

    # Send the formatted prompt to the initialized LLM and get the response.
    response = llm.invoke(prompt)
    # Return only the content part of the LLM's response object.
    return response.content