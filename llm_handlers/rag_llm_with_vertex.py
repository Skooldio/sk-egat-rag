# Import necessary libraries
import streamlit as st # For accessing Streamlit secrets and UI elements
import os # For interacting with the operating system (not heavily used here)
from langchain_community.document_loaders import PyPDFLoader # PDF loader (not used directly)
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text splitter (not used directly)
from langchain_google_vertexai import VertexAIEmbeddings # Vertex AI Embeddings (not used directly)
from langchain_core.vectorstores import InMemoryVectorStore # In-memory store (not used directly)
from langchain_openai import OpenAIEmbeddings # OpenAI Embeddings (not used here)
from google.oauth2 import service_account # For Google Cloud credentials
from langchain.chat_models import init_chat_model # Function to initialize chat models
from langchain_core.prompts import PromptTemplate # For creating prompt templates
from langchain_google_community import (
    VertexAISearchRetriever, # Retriever specifically for Google Cloud Vertex AI Search
)

# Load Google Cloud service account credentials from a local JSON file.
# Required for authenticating with Vertex AI services (LLM and Search).
egat_service_account = service_account.Credentials.from_service_account_file(filename="./service_account.json")

# Initialize the Language Model (LLM) using Google Vertex AI.
# Specifies the Gemini model and provides the necessary credentials.
llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai", credentials=egat_service_account)
# llm = init_chat_model("gpt-4o", model_provider="openai") # Alternative: Initialize OpenAI model (commented out)

# Define the prompt template string for the LLM.
# This template guides the LLM to act as an assistant for EGAT procurement regulations,
# using the retrieved documents as the sole source, adhering to specific rules and formatting.
# Placeholders {retrieved_documents} and {user_prompt} will be filled later.
template="""
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
# Replace with your specific Google Cloud project and Vertex AI Search details.
PROJECT_ID = "skooldio-egat-rag-workshop"  # Your Google Cloud Project ID
LOCATION_ID = "global"  # Location of your Vertex AI Search data store (e.g., "us", "eu", "global")
SEARCH_ENGINE_ID = "egat-prompt-rag_1743063246416"  # Your Vertex AI Search App ID (Engine ID) - Note: May not be directly used by retriever if Data Store ID is sufficient
DATA_STORE_ID = "egat-document-store_1743063327319" # Your Vertex AI Search Data Store ID
# --- End Vertex AI Search Configuration ---

# Define the function to handle LLM response generation using RAG with Vertex AI Search.
def llm_response(messages):
    """
    Generates a response using RAG: retrieves relevant documents from Vertex AI Search
    based on the user prompt, formats them into a prompt using the template,
    sends the prompt to the LLM (Gemini), and returns the response.

    Args:
        messages (list): A list of message dictionaries from the chat history.

    Returns:
        str: The content of the LLM's generated response.
    """
    # Get the latest user message content.
    user_prompt = messages[-1]["content"]
    print ("user_prompt", user_prompt) # Print for debugging

    # Initialize the Vertex AI Search Retriever.
    # Connects to the specified data store in your Google Cloud project.
    retriever = VertexAISearchRetriever(
        credentials=egat_service_account, # Pass the loaded credentials
        project_id=PROJECT_ID, # Specify the Google Cloud Project ID
        location_id=LOCATION_ID, # Specify the location of the data store
        data_store_id=DATA_STORE_ID, # Specify the ID of the data store to search within
        max_documents=10, # Set the maximum number of documents to retrieve
        engine_data_type=0, # Corresponds to 'unstructured' data store type
        get_extractive_answers=True, # Enable feature to get direct answers from search results if available
    )
    # Retrieve relevant documents from Vertex AI Search based on the user prompt.
    retrieved_docs = retriever.invoke(user_prompt)

    # Display the retrieved documents in the Streamlit UI within an expander.
    with st.expander("Retrieved Documents"):
        st.write(retrieved_docs)

    # Create a PromptTemplate object from the defined template string.
    custom_template = PromptTemplate.from_template(template)

    # Helper function to replace Thai numerals (๐-๙) with Arabic numerals (0-9).
    # Useful for consistency or if the LLM handles Arabic numerals better.
    def replace_thai_no(text):
        return text.replace("๐", "0").replace("๑", "1").replace("๒", "2").replace("๓", "3").replace("๔", "4").replace("๕", "5").replace("๖", "6").replace("๗", "7").replace("๘", "8").replace("๙", "9")

    # Format the retrieved documents into a single string for the prompt context.
    # Joins the 'page_content' of each document, applying the numeral replacement.
    docs_content = "\n\n".join(replace_thai_no(doc.page_content) for doc in retrieved_docs)
    # st.write(docs_content) # Optional: Display the formatted context in Streamlit

    # Create the final prompt by invoking the template with the user prompt and formatted documents.
    prompt = custom_template.invoke({"user_prompt": user_prompt, "retrieved_documents":docs_content})

    # Print the final prompt sent to the LLM for debugging.
    print ("Final Prompt", prompt)

    # Invoke the initialized LLM (Gemini) with the final prompt.
    response = llm.invoke(prompt)
    # Return the content part of the LLM's response.
    return response.content