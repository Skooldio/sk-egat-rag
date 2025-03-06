@echo off
REM filepath: $CMD/start.bat

REM Create virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Create .streamlit directory if it doesn't exist
if not exist .streamlit mkdir .streamlit

REM Create a basic secrets.toml file
(
echo # This is the Streamlit secrets file
echo.
echo [llm]
echo openai_api_key = ""
) > .streamlit\secrets.toml

echo Setup complete!