# Import necessary libraries
import streamlit as st # For accessing Streamlit secrets and UI elements
import os # For interacting with the operating system (e.g., environment variables)
from langchain_community.document_loaders import PyPDFLoader # PDF loader (though not used directly in this RAG flow, might be for future use)
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text splitter (similarly, not used directly here)
from langchain_google_vertexai import VertexAIEmbeddings # Vertex AI Embeddings (not used in this Azure flow)
from langchain_core.vectorstores import InMemoryVectorStore # In-memory vector store (not used in this Azure flow)
from langchain_openai import OpenAIEmbeddings # OpenAI Embeddings (not used directly, as Azure Search handles embeddings)
from google.oauth2 import service_account # Google credentials (not used in this Azure/OpenAI flow)
from langchain.chat_models import init_chat_model # Function to initialize chat models
from langchain_core.prompts import PromptTemplate # For creating prompt templates
from langchain_community.retrievers import AzureAISearchRetriever # Retriever specifically for Azure AI Search

# Set the OpenAI API key from Streamlit secrets. Required for the OpenAI LLM.
os.environ["OPENAI_API_KEY"] = st.secrets.llm.openai_api_key

# Initialize the Language Model (LLM).
# Option 1 (Commented out): Use Google's Gemini via Vertex AI.
# llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai", credentials=egat_service_account)
# Option 2 (Active): Use OpenAI's GPT-4o model.
llm = init_chat_model("gpt-4o", model_provider="openai")

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

# Configure Azure AI Search connection details using environment variables.
# These should ideally be set securely (e.g., via Streamlit secrets or system environment variables).
os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "" # Name of your Azure AI Search service
os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = "" # Name of the specific index to query
# WARNING: Hardcoding API keys is insecure. Use st.secrets or environment variables in production.
os.environ["AZURE_AI_SEARCH_API_KEY"] = (
    ""
) # API Key for accessing the Azure AI Search service

# Define the function to handle LLM response generation using RAG with Azure AI Search.
def llm_response(messages):
    """
    Generates a response using RAG: retrieves relevant documents from Azure AI Search
    based on the user prompt, formats them into a prompt using the template,
    sends the prompt to the LLM (GPT-4o), and returns the response.

    Args:
        messages (list): A list of message dictionaries from the chat history.

    Returns:
        str: The content of the LLM's generated response.
    """
    # Get the latest user message content.
    user_prompt = messages[-1]["content"]
    print ("user_prompt", user_prompt) # Print for debugging

    # Initialize the Azure AI Search Retriever.
    # Specifies the field containing the document content ('chunk') and the index name.
    # The API version might need adjustment based on the Azure service features used.
    retriever = AzureAISearchRetriever(
        content_key="chunk", # The field in the Azure index containing the text content
        index_name="", # The target index name
        api_version="2024-11-01-preview", # Azure Search API version
    )

    # Retrieve relevant documents from Azure AI Search based on the user prompt.
    retrieved_docs = retriever.invoke(user_prompt)

    # Display the retrieved documents in the Streamlit UI within an expander.
    with st.expander("Retrieved Documents"):
        st.write(retrieved_docs)

    # Create a PromptTemplate object from the defined template string.
    custom_template = PromptTemplate.from_template(template)

    # Helper function to replace Thai numerals (๐-๙) with Arabic numerals (0-9).
    # This might be needed if the LLM handles Arabic numerals better or for consistency.
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

    # Invoke the initialized LLM (GPT-4o) with the final prompt.
    response = llm.invoke(prompt)
    # Return the content part of the LLM's response.
    return response.content