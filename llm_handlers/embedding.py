# Import necessary libraries
from langchain_google_vertexai import VertexAIEmbeddings # For generating text embeddings using Google Vertex AI
import streamlit as st # For creating the web application interface

# Import evaluation utilities from LangChain
from langchain.evaluation import load_evaluator # Function to load pre-defined evaluators
from langchain.evaluation import EmbeddingDistance # Enum for embedding distance metrics (e.g., COSINE, EUCLIDEAN)
from langchain.evaluation import StringDistance # Enum for string distance metrics (e.g., LEVENSHTEIN)

# Initialize the Vertex AI embedding model.
# Replace 'text-embedding-004' with your desired model if needed.
# Note: This assumes necessary authentication (e.g., service account) is handled elsewhere or via environment setup.
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Load an evaluator for calculating the distance between embedding vectors.
# This uses the initialized 'embeddings' object to generate vectors before calculating distance.
# The default distance metric is COSINE, EUCLIDEAN is commented out as an alternative.
vector_evaluator = load_evaluator(
    "embedding_distance", # Specify the type of evaluator
    embeddings=embeddings, # Provide the embedding model to use
    # distance_metric=EmbeddingDistance.EUCLIDEAN, # Optionally specify a different distance metric
)
# Load an evaluator for calculating the distance between strings (text).
# This calculates lexical distance, not semantic distance based on embeddings.
# LEVENSHTEIN distance measures the minimum number of single-character edits required to change one word into the other.
string_evaluator = load_evaluator(
    "string_distance", distance_metric=StringDistance.LEVENSHTEIN
)

# Define a function to demonstrate embeddings and distances within the Streamlit app.
def show_embeddings():
    """
    Displays example sentences, their embeddings, and the calculated
    vector and string distances between them using Streamlit components.
    """
    # Define two sample sentences for comparison.
    sentence1 = "Cats are great pets"
    sentence2 = "Cats make good companions"

    # Display the sentences in the Streamlit app.
    st.write("Sentence 1:", sentence1)
    st.write("Sentence 2:", sentence2)

    # Use an expander widget to optionally show the raw embedding vectors.
    with st.expander("See Embeddings"):
        # Generate and display the embedding vector for sentence 1.
        st.write("Embedding 1:", embeddings.embed_query(sentence1))
        # Generate and display the embedding vector for sentence 2.
        st.write("Embedding 2:", embeddings.embed_query(sentence2))

    # Calculate and display the embedding vector distance between the two sentences.
    # The evaluator handles embedding generation and distance calculation.
    st.write(
        "Vector Distance",
        vector_evaluator.evaluate_strings(prediction=sentence1, reference=sentence2),
    )
    # Calculate and display the string (Levenshtein) distance between the two sentences.
    st.write(
        "String Distance",
        string_evaluator.evaluate_strings(prediction=sentence1, reference=sentence2),
    )