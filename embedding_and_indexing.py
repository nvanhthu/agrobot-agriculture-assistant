#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pickle
from openai import OpenAI
from typing import List
from PyPDF2 import PdfReader
import numpy as np
import faiss
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client (make sure OPENAI_API_KEY is set in the environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

CHUNKS_SAVE_PATH = "Processed_Data/chunks.pkl"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536  # Dimension for ada-002 embeddings
MAX_TOKENS = 8000     # Keep below model max context length

# Initialize tiktoken encoding for the model
ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)

def load_text_files(folder_path):
    """
    Load all text files from a folder and return a list of raw texts.
    """
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                chunks.append(text)
    return chunks

def chunk_text(text, max_tokens=MAX_TOKENS):
    """
    Split a single large text into smaller chunks, each <= max_tokens tokens.
    """
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODING.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks

def chunk_all_texts(texts, max_tokens=MAX_TOKENS):
    """
    Take a list of texts and return a flattened list of token-safe chunks.
    """
    all_chunks = []
    for text in texts:
        text_chunks = chunk_text(text, max_tokens)
        all_chunks.extend(text_chunks)
    return all_chunks

def get_embeddings(texts):
    """
    Get embeddings from OpenAI API for a list of texts.
    """
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding_vector = np.array(response.data[0].embedding, dtype=np.float32)
        embeddings.append(embedding_vector)
    return embeddings

def save_faiss_index(embeddings, file_path="Processed_Data/faiss_index.bin"):
    """
    Save FAISS index built from embeddings.
    """
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(np.vstack(embeddings))
    faiss.write_index(index, file_path)
    print(f"FAISS index saved at: {file_path}")

def main():
    folder_path = "Processed_Data"

    print("Loading text files...")
    raw_texts = load_text_files(folder_path)
    print(f"Loaded {len(raw_texts)} raw text files.")

    print("Splitting texts into token-safe chunks...")
    text_chunks = chunk_all_texts(raw_texts, MAX_TOKENS)
    print(f"Number of chunks after splitting: {len(text_chunks)}")

    print("Creating embeddings for each chunk...")
    embeddings = get_embeddings(text_chunks)
    print("Embeddings created.")

    print("Saving FAISS index...")
    save_faiss_index(embeddings)
    
    # Save chunks for retrieval
    with open(CHUNKS_SAVE_PATH, "wb") as f:
        pickle.dump(text_chunks, f)
    print(f"Chunks saved at: {CHUNKS_SAVE_PATH}")

if __name__ == "__main__":
    main()


# In[ ]:




