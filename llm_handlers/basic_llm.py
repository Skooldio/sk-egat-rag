import streamlit as st


def llm_response(messages):
    return messages[len(messages) - 1]["content"]
