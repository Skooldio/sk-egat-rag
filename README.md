# **RAG (Retrieval Augmented Generation) Application**

This repository contains a Streamlit-based chatbot application that uses Retrieval Augmented Generation (RAG) to provide informative responses based on document retrieval.

## **Setup Instructions**

### **Windows Setup**

1. Make sure you have Python installed on your system.
2. Open Command Prompt or PowerShell.
3. Navigate to the project directory.
4. Run the setup script:

```bash
start.bat
```

This script will:

- Create a virtual environment called venv
- Activate the virtual environment
- Install the required dependencies from requirements.txt
- Create a .streamlit directory with a basic secrets.toml file

### **Mac/Linux Setup**

1. Make sure you have Python installed on your system.
2. Open Terminal.
3. Navigate to the project directory.
4. Make the script executable and run it:

```bash
chmod +x start.sh
./start.sh
```

This script will:

- Create a virtual environment called venv
- Activate the virtual environment
- Install the required dependencies from requirements.txt
- Create a .streamlit directory with a basic secrets.toml file

## **Running the Application**

After completing the setup, you can run the Streamlit application:

1. Ensure your virtual environment is activated:
    - Windows: `venv\Scripts\activate`
    - Mac/Linux: `source venv/bin/activate`
2. Run the Streamlit application:

```bash
streamlit run app.py
```
3. The application will be available in your web browser at `http://localhost:8501`

## **Configuration**

- API keys for OpenAI and other services can be configured in secrets.toml
- Google service account credentials are stored in service_account.json

## **Features**

This application demonstrates various RAG implementations including:

- Basic LLM integration
- Vector search using LangChain
- Google Vertex AI integration
- Azure Search AI integration
- RAG evaluation techniques

## Repository Structure and File Responsibilities
├── llm_handlers/
│   ├── basic_llm.py             # Basic LLM interaction handler
│   ├── embedding.py             # Handles text embeddings and comparisons
│   ├── rag_llm.py               # RAG implementation using local vector store
│   ├── rag_llm_with_azure.py    # RAG implementation using Azure AI Search
│   ├── rag_llm_with_vertex.py   # RAG implementation using Google Vertex AI Search
│   └── rag_llm_with_vertex_with_evals.py # RAG with Vertex AI Search + Trulens evaluation
├── .streamlit/
│   └── secrets.toml             # Streamlit secrets configuration (created by setup script)
├── app.py                       # Main Streamlit application file
├── README.md                    # This file - provides information about the repository
├── Regulation_389.pdf           # EGAT Regulation document (knowledge base) 
├── requirements.txt             # Python dependencies for the project 
├── service_account.json         # Google Cloud service account credentials 
├── start.bat                    # Setup and run script for Windows 
└── start.sh                     # Setup and run script for Mac/Linux 

### Core Files

* **`app.py`**: The entry point for the Streamlit web application. It defines the user interface for the chatbot, manages the chat history, and calls the appropriate LLM handler to generate responses based on user input. [cite: 1]
* **`Regulation_389.pdf`**: The source document containing EGAT's procurement regulations (ฉบับที่ ๓๘๙). This PDF is used as the primary knowledge base for the RAG system to retrieve information from when answering user queries. [cite: 2]
* **`requirements.txt`**: Lists all necessary Python packages (like `streamlit`, `langchain`, `google-cloud-aiplatform`, etc.) that need to be installed for the application to run correctly. [cite: 1]
* **`start.sh` / `start.bat`**: Setup scripts for different operating systems (Mac/Linux and Windows, respectively). They automate the process of creating a Python virtual environment, installing the required dependencies from `requirements.txt`, and creating a basic `.streamlit/secrets.toml` file for API keys. [cite: 1]

### LLM Handlers (`llm_handlers/`)

This directory contains different modules for handling the logic of interacting with language models and implementing the RAG pipeline:

* **`basic_llm.py`**: Provides a simple interface to interact directly with an LLM (like Google's Gemini or OpenAI's GPT models) without retrieval augmentation. It includes a basic prompt template. [cite: 1]
* **`embedding.py`**: Focuses on text embeddings. It uses `VertexAIEmbeddings` to generate vector representations of text and includes functions for comparing embeddings using different distance metrics, useful for understanding semantic similarity. [cite: 1]
* **`rag_llm.py`**: Implements the core RAG logic using LangChain components. It loads `Regulation_389.pdf`[cite: 2], splits it into manageable chunks (parent and child documents), generates embeddings using `VertexAIEmbeddings`, stores them in an `InMemoryVectorStore`, and uses a `ParentDocumentRetriever` to find relevant document chunks based on the user query. These chunks are then passed to a `ChatVertexAI` model (Gemini) along with a specific prompt template to generate an answer grounded in the document. [cite: 1]
* **`rag_llm_with_vertex.py`**: An alternative RAG implementation that utilizes Google Cloud Vertex AI Search as the retriever. Instead of processing and storing the PDF locally, it queries a pre-configured Vertex AI Search data store (containing the indexed `Regulation_389.pdf` [cite: 2]) to find relevant information. It uses the same Gemini LLM and prompt template for generation. [cite: 1]
* **`rag_llm_with_azure.py`**: Another RAG variant, this time using Azure AI Search as the retriever. It connects to an Azure AI Search index to fetch relevant document chunks and uses an OpenAI model (GPT-4o) for generating the final answer. [cite: 1]
* **`rag_llm_with_vertex_with_evals.py`**: Extends `rag_llm_with_vertex.py` by integrating Trulens for RAG pipeline evaluation. It sets up feedback functions (Context Relevance, Groundedness, Answer Relevance) using an OpenAI provider and runs sample queries through the RAG chain to record and potentially evaluate the quality of the retrieval and generation steps. [cite: 1]

*(Note: The specific handler used by `app.py` might be commented/uncommented within `app.py` itself to switch between different RAG implementations or the basic LLM.)* [cite: 1]
