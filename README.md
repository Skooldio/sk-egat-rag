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
