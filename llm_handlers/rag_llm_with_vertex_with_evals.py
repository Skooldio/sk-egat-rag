“# Import necessary libraries
import streamlit as st # For potential Streamlit integration (though not directly used in this evaluation script)
import os # For interacting with the operating system (e.g., environment variables)
from langchain_community.document_loaders import PyPDFLoader # PDF Loader (not used here)
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text Splitter (not used here)
from langchain_google_vertexai import VertexAIEmbeddings # Vertex AI Embeddings (not used directly here, handled by Vertex AI Search)
from langchain_core.vectorstores import InMemoryVectorStore # In-memory store (not used here)
from langchain_openai import OpenAIEmbeddings # OpenAI Embeddings (not used here)
from google.oauth2 import service_account # For Google Cloud credentials
from langchain.chat_models import init_chat_model # Function to initialize chat models
from langchain_core.prompts import PromptTemplate # For creating prompt templates
from langchain_google_community import (
    VertexAISearchRetriever, # Retriever for Google Cloud Vertex AI Search
)
from langchain.schema import StrOutputParser # Simple output parser for LangChain chains
from langchain_core.runnables import RunnablePassthrough # Passthrough component for LangChain Expression Language (LCEL)

# Load Google Cloud service account credentials from a local JSON file.
egat_service_account = service_account.Credentials.from_service_account_file(
    filename="./service_account.json"
)

# Initialize the Google Gemini LLM via Vertex AI using the loaded credentials.
llm = init_chat_model(
    "gemini-2.0-flash-001", # Specify the Gemini model version
    model_provider="google_vertexai", # Indicate the provider
    credentials=egat_service_account, # Pass the credentials
)
# llm = init_chat_model("gpt-4o", model_provider="openai") # Alternative: Initialize OpenAI model (commented out)

# Define the prompt template string for the LLM.
# This template guides the LLM to answer questions based on retrieved EGAT documents.
template = """
คุณคือผู้ช่วยตอบคำถามเกี่ยวกับระเบียบการจัดซื้อจัดจ้างของ กฟผ. โดยใช้เอกสารแนบจาก Retrieved Documents ที่ให้ไว้เป็นแหล่งข้อมูลอ้างอิงหลักเพียงแหล่งเดียว

📌 กฎการตอบ:
1. ให้ตอบโดยอ้างอิงจากเนื้อหาในเอกสารที่ให้ไว้เท่านั้น
2. ห้ามแต่งเติมเนื้อหานอกเหนือจากเอกสาร
3. หากไม่มีข้อมูลในเอกสาร ให้ตอบว่า “ไม่พบข้อมูลในเอกสารที่ให้ไว้”
4. ทุกครั้งที่ตอบ ให้ระบุแหล่งที่มาหรือการอ้างอิง เช่น เลขหน้า, หมวด, ข้อ หรือหัวข้อ
5. **ตอบให้เหมือนเจ้าหน้าที่ที่รู้จริงตอบแบบเป็นกันเอง ไม่ใช้ภาษาทางการมากเกินไป**
6. **ห้ามตอบเวิ่นเว้อ หรืออธิบายเกินกว่าที่คำถามต้องการ**
7. **ตอบตรงประเด็น และสั้นเท่าที่จำเป็นให้เข้าใจง่าย**

📌 รูปแบบการตอบ:
- เริ่มด้วยคำตอบสั้น กระชับ ตรงประเด็น
- ต่อท้ายด้วยบรรทัดใหม่ว่า:

  _อ้างอิงจาก: หน้า XX, หมวด X, ข้อ X_

📌 ตัวอย่างการตอบ:
“วงเงินไม่เกิน 100,000 บาท ใช้วิธีเฉพาะเจาะจงได้ แต่ต้องมีเหตุผลและอนุมัติจากผู้มีอำนาจ”

_อ้างอิงจาก: หน้า 12, หมวด 2, ข้อ 5.1_

กรุณารอรับคำถามจากผู้ใช้งาน


## Retrieved Documents !!!!:
{retrieved_documents}

## คำถามจากผู้ใช้งาน !!!!:
{user_prompt}
"""

# --- Vertex AI Search Configuration ---
PROJECT_ID = ""  # Your Google Cloud Project ID
LOCATION_ID = "global"  # Location of your Vertex AI Search data store
SEARCH_ENGINE_ID = ""  # Your Vertex AI Search App ID (Engine ID)
DATA_STORE_ID = "" # Your Vertex AI Search Data Store ID
# --- End Vertex AI Search Configuration ---

# Helper function to format the retrieved documents into a single string.
def format_docs(docs):
    """Concatenates the page_content of retrieved LangChain Documents."""
    return "\n\n".join(doc.page_content for doc in docs)

# Create a PromptTemplate object from the template string.
custom_template = PromptTemplate.from_template(template)

# Initialize the Vertex AI Search Retriever.
retriever = VertexAISearchRetriever(
    credentials=egat_service_account, # Pass credentials
    project_id=PROJECT_ID, # Specify Project ID
    location_id=LOCATION_ID, # Specify Data Store Location
    data_store_id=DATA_STORE_ID, # Specify Data Store ID
    max_documents=10, # Maximum number of documents to retrieve
    engine_data_type=0,  # Corresponds to 'unstructured' data store type
    get_extractive_answers=True, # Attempt to get direct answers from the search results
)

# Define the RAG chain using LangChain Expression Language (LCEL).
rag_chain = (
    {
        # The 'retrieved_documents' key will hold the formatted content from the retriever.
        "retrieved_documents": retriever | format_docs, # Pipe output of retriever to format_docs
        # The 'user_prompt' key passes the original input through.
        "user_prompt": RunnablePassthrough(), # Pass the input query directly
    }
    | custom_template # Pipe the dictionary to the prompt template
    | llm # Pipe the formatted prompt to the LLM
    | StrOutputParser() # Parse the LLM output to a string
)

# (Commented out) Original function for getting LLM response, now replaced by direct chain invocation for evaluation.
# def llm_response(messages):
#     user_prompt = messages[-1]["content"]
#     return rag_chain.invoke(user_prompt)

# --- Trulens Evaluation Setup ---
from trulens.core.session import TruSession # Import TruSession for managing evaluation sessions

# Initialize or get the default Trulens session.
session = TruSession()
# Reset the database (clears previous evaluation records for this session).
session.reset_database()

from trulens.dashboard import run_dashboard # Import function to run the Trulens dashboard

# Start the Trulens dashboard (runs as a separate process/web server).
# 'force=True' restarts the dashboard if it's already running.
run_dashboard(session=session, force=True)

# Set OpenAI API key - Required for the Trulens OpenAI provider used for evaluations.
# WARNING: Hardcoding API keys is insecure. Use secrets management in production.
os.environ["OPENAI_API_KEY"] = ""

from trulens.providers.openai import OpenAI # Import the Trulens OpenAI provider

# Initialize the OpenAI provider (used for evaluation metrics, not the main RAG LLM).
provider = OpenAI()

from trulens.apps.langchain import TruChain # Import TruChain for instrumenting LangChain apps

# Select the part of the RAG chain that represents the retrieved context.
# This allows Trulens to evaluate the relevance of the documents retrieved by 'retriever'.
context = TruChain.select_context(rag_chain)

from trulens.core import Feedback # Import the Feedback class for defining evaluation metrics
import numpy as np # Import numpy for aggregation functions (like mean)

# Define a feedback function for Context Relevance.
# Uses the OpenAI provider to evaluate if the retrieved context is relevant to the input query.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") # Use OpenAI's CoT method
    .on_input() # Operates on the main input query
    .on(context) # Evaluates the selected context
    .aggregate(np.mean) # Averages the scores if multiple contexts are evaluated
)

# Define a feedback function for Groundedness.
# Uses the OpenAI provider to check if the LLM's response is supported by the retrieved context.
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") # Use OpenAI's CoT method
    .on(context.collect())  # Collect context chunks into a list before evaluation
    .on_output() # Evaluates the final output response
)

# Define a feedback function for Answer Relevance.
# Uses the OpenAI provider to evaluate if the LLM's response is relevant to the input query.
f_answer_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance" # Use OpenAI's CoT method
).on_input_output() # Operates on both the input query and the final output response
# --- End Trulens Evaluation Setup ---

# Instrument the LangChain RAG chain with Trulens for evaluation.
tru_recorder = TruChain(
    rag_chain, # The LangChain chain to evaluate
    app_name="ChatApplication", # Name for the application in Trulens dashboard
    app_version="Chain1", # Version identifier
    feedbacks=[f_context_relevance, f_groundedness, f_answer_relevance], # List of feedback functions to apply
)

# Run example queries through the instrumented chain.
# The 'with tru_recorder as recording:' block ensures that each invocation is recorded by Trulens.
with tru_recorder as recording:
    # Invoke the chain with several test prompts.
    # Each invocation will be evaluated using the defined feedback functions.
    # Results can be viewed in the Trulens dashboard.
    rag_chain.invoke("การจัดซื้อจัดจ้างของ กฟผ. มีกี่ประเภท")
    rag_chain.invoke("การจัดซื้อจัดจ้างของ กฟผ. มีกี่วิธี")
    rag_chain.invoke("การจัดซื้อจัดจ้างวงเงินไม่เกิน 100,000 บาท ใช้วิธีไหนได้บ้าง")
    rag_chain.invoke("ข้อบังคับนี้บังคับใช้เมื่อไหร่")
    rag_chain.invoke("การจัดซื้อจัดจ้าง มีกี่ประเภท")
    rag_chain.invoke("การจัดซื้อจัดจ้าง มีกี่วิธี")
    rag_chain.invoke("การจัดซื้อจัดจ้างวงเงินไม่เกิน 500,000 บาท ใช้วิธีไหนได้บ้าง")