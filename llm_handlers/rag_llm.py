import streamlit as st  # Import the Streamlit library for creating web applications
import os  # Import the os module for interacting with the operating system
from langchain_community.document_loaders import PyPDFLoader  # Import PyPDFLoader to load PDF documents

from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import RecursiveCharacterTextSplitter for splitting text recursively
from langchain_google_vertexai import VertexAIEmbeddings  # Import VertexAIEmbeddings for creating embeddings using Google Vertex AI
from langchain_core.vectorstores import InMemoryVectorStore  # Import InMemoryVectorStore to store vectors in memory
from langchain_openai import OpenAIEmbeddings  # Import OpenAIEmbeddings for creating embeddings using OpenAI (commented out)
from google.oauth2 import service_account  # Import service_account for handling Google service account credentials
# from langchain.chat_models import init_chat_model # Deprecated, use specific chat model classes
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate for creating prompt templates
from langchain.retrievers import ParentDocumentRetriever  # Import ParentDocumentRetriever for retrieving documents based on parent-child relationships
from langchain.storage import InMemoryStore  # Import InMemoryStore for storing documents in memory

# from langchain_experimental.text_splitter import SemanticChunker # Import SemanticChunker (commented out)

# Load Google service account credentials from a JSON file
egat_service_account = service_account.Credentials.from_service_account_file(
    filename="./service_account.json"
)

file_path = "./Regulation_389.pdf"  # Define the path to the PDF document
os.environ["OPENAI_API_KEY"] = st.secrets.llm.openai_api_key  # Set the OpenAI API key from Streamlit secrets (even though OpenAIEmbeddings is commented out, this is likely for an alternative LLM)

# Initialize Vertex AI embeddings
embeddings = VertexAIEmbeddings(
    model="text-embedding-005", credentials=egat_service_account
)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # Initialize OpenAI embeddings (commented out)

# Initialize in-memory vector store and document store
vector_store = InMemoryVectorStore(embeddings)
store = InMemoryStore()

# Initialize text splitters for parent and child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Initialize the ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,  # The vector store for storing child document embeddings
    docstore=store,  # The document store for storing parent documents
    parent_splitter=parent_splitter,  # The splitter for creating parent documents
    child_splitter=child_splitter,  # The splitter for creating child documents
)


def load_data():
    """
    Loads the PDF document, splits it into parent and child documents,
    and adds them to the retriever's vector and document stores.
    """
    st.write("Loading data...")  # Display a message indicating data loading
    loader = PyPDFLoader(file_path)  # Create a PDF loader instance
    docs = loader.load()  # Load the documents from the PDF

    # Function to replace Thai numerals with Arabic numerals (commented out)
    # def replace_thai_no(text):
    #       return text.replace("๐", "0").replace("๑", "1").replace("๒", "2").replace("๓", "3").replace("๔", "4").replace("๕", "5").replace("๖", "6").replace("๗", "7").replace("๘", "8").replace("๙", "9")

    st.write("Document Page", len(docs))  # Display the number of loaded document pages
    retriever.add_documents(docs)  # Add the loaded documents to the retriever

    # Perform a similarity search on the vector store with a sample query
    sub_docs = vector_store.similarity_search("การจัดซื้อมีกี่วิธี", k=3)
    st.write("Sub Documents", sub_docs)  # Display the results of the similarity search

    # Retrieve documents using the parent document retriever with a sample query
    retrievered_docs = retriever.invoke("การจัดซื้อมีกี่วิธี")
    st.write("Retrieved Documents", retrievered_docs)  # Display the retrieved documents


# Initialize the chat model using Google Vertex AI
from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(
    model="gemini-2.0-flash-001",
    credentials=egat_service_account,
)
# llm = init_chat_model("gpt-4o", model_provider="openai") # Initialize OpenAI chat model (commented out)

# Define the prompt template for the language model
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


def llm_response(messages):
    """
    Generates a response from the language model based on the user's prompt
    and retrieved documents.

    Args:
        messages (list): A list of message dictionaries, typically from a chat history.

    Returns:
        str: The content of the language model's response.
    """
    user_prompt = messages[-1]["content"]  # Get the latest user prompt from the messages
    print("user_prompt", user_prompt)  # Print the user prompt for debugging

    # retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Convert vector store to a retriever (commented out)
    retrieved_docs = retriever.invoke(user_prompt)  # Retrieve relevant documents based on the user prompt

    with st.expander("Retrieved Documents"):
        st.write(retrieved_docs)  # Display the retrieved documents in an expandable section

    custom_template = PromptTemplate.from_template(template)  # Create a PromptTemplate from the defined template

    def replace_thai_no(text):
        """Replaces Thai numerals with Arabic numerals in a given text."""
        return (
            text.replace("๐", "0")
            .replace("๑", "1")
            .replace("๒", "2")
            .replace("๓", "3")
            .replace("๔", "4")
            .replace("๕", "5")
            .replace("๖", "6")
            .replace("๗", "7")
            .replace("๘", "8")
            .replace("๙", "9")
        )

    # Concatenate the content of the retrieved documents, replacing Thai numerals
    docs_content = "\n\n".join(
        replace_thai_no(doc.page_content) for doc in retrieved_docs
    )
    # st.write(docs_content) # Display the concatenated document content (commented out)

    # Invoke the prompt template with the user prompt and retrieved document content
    prompt = custom_template.invoke(
        {"user_prompt": user_prompt, "retrieved_documents": docs_content}
    )

    print("Final Prompt", prompt)  # Print the final prompt sent to the LLM for debugging

    response = llm.invoke(prompt)  # Invoke the language model with the final prompt
    return response.content  # Return the content of the language model's response
