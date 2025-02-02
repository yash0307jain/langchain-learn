import streamlit as st

# Set page config
st.set_page_config(page_title="Langchain Streamlit Basic Chatbot", page_icon="🙏")
st.title("Chatbot using Streamlit and Langchain")

prompt = st.chat_input("Say Something!")
if prompt:
    st.write(f"User has send the following {prompt}")
