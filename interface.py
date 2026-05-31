import streamlit as st
from rag_utils import (
    load_text_files,
    split_text,
    load_faiss_index,
    retrieve_similar_chunks,
    generate_answer,
)
import os

@st.cache_resource
def load_resources():
    folder_path = "Processed_Data"
    raw_texts = load_text_files(folder_path)
    text_chunks = []
    for text in raw_texts:
        text_chunks.extend(split_text(text))
    index = load_faiss_index("Processed_Data/faiss_index.bin")
    return text_chunks, index

text_chunks, index = load_resources()

st.title("🤖 Agrobot - Farm Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

def submit():
    user_input = st.session_state.user_input.strip()
    if user_input:
        # Retrieve top chunks relevant to query
        relevant_chunks = retrieve_similar_chunks(user_input, text_chunks, index)
        # Generate answer with context
        answer = generate_answer(user_input, relevant_chunks)
        # Append to chat history
        st.session_state.history.append({"user": user_input, "bot": answer})
        st.session_state.user_input = ""  # clear input box

# Display chat history
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Agrobot:** {chat['bot']}")
    st.markdown("---")

# Input box that submits on Enter
st.text_input(
    "Ask your question:",
    key="user_input",
    on_change=submit,
    placeholder="Type your question and press Enter...",
)
