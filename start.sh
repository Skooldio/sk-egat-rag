#!/bin/bash

python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create a basic secrets.toml file
cat > .streamlit/secrets.toml << 'EOF'
# This is the Streamlit secrets file

[llm]
openai_api_key = ""

EOF
