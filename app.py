# Import the Streamlit library for creating the web application interface
import streamlit as st

# --- LLM Handler Imports ---
# Import functions from different modules in the 'llm_handlers' directory.
# Only one of these should typically be active at a time, determining the chatbot's backend logic.

# Option 1: Basic LLM without RAG
# from llm_handlers.basic_llm import llm_response

# Option 2: Embedding demonstration (usually displayed in sidebar, not for chat response)
# from llm_handlers.embedding import show_embeddings

# Option 3: RAG using local PDF loading, splitting, and in-memory vector store (ParentDocumentRetriever)
# Requires uncommenting 'load_data()' in the sidebar as well.
# from llm_handlers.rag_llm import load_data, llm_response

# Option 4: RAG using Vertex AI Search for retrieval and Trulens for evaluation (Currently Active)
# from llm_handlers.rag_llm_with_vertex_with_evals import llm_response

# Option 5: RAG using Azure AI Search (Requires uncommenting and commenting out others)
# from llm_handlers.rag_llm_with_azure import llm_response

# Option 6: RAG using Vertex AI Search (without Trulens evaluation)
from llm_handlers.rag_llm_with_vertex import llm_response
# --- End LLM Handler Imports ---


# Set the title of the Streamlit web application page.
st.title("💬 Chatbot") #

# --- Sidebar Setup ---
# Create a sidebar section for controls or additional information.
with st.sidebar:
    # Add some text to the sidebar.
    st.write("This is a playground sidebar.") #
    # (Commented out) Call function to display embedding examples in the sidebar.
    # show_embeddings()
    # (Commented out) Call function to load data for the local RAG implementation ('rag_llm.py').
    # This should be called only if using 'rag_llm.py'.
    # load_data()
# --- End Sidebar Setup ---


# --- Chat Interface Logic ---

# Initialize the chat message history in Streamlit's session state if it doesn't exist.
# Session state persists data across reruns of the script (e.g., when user interacts).
if "messages" not in st.session_state:
    # Start with a default assistant message.
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ] #

# Display the existing chat messages from the session state.
# Loop through each message stored in the session state.
for msg in st.session_state.messages:
    # Display the message using Streamlit's chat message component, assigning the correct role (user/assistant).
    st.chat_message(msg["role"]).write(msg["content"]) #

# Get user input using Streamlit's chat input widget.
# The 'if prompt := ...' syntax assigns the input to 'prompt' if the user enters something.
if prompt := st.chat_input():
    # Append the user's message to the session state history.
    st.session_state.messages.append({"role": "user", "content": prompt}) #
    # Display the user's message immediately in the chat interface.
    st.chat_message("user").write(prompt) #

    # Call the selected LLM response function (imported earlier)
    # Pass the entire message history to the function.
    msg = llm_response(st.session_state.messages) #

    # Append the assistant's response to the session state history.
    st.session_state.messages.append({"role": "assistant", "content": msg}) #
    # Display the assistant's message in the chat interface.
    st.chat_message("assistant").write(msg) #
# --- End Chat Interface Logic ---