import streamlit as st
from openai import OpenAI

openai_api_key = st.secrets.llm.openai_api_key
client = OpenAI(api_key=openai_api_key)


def llm_response(messages):
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content
