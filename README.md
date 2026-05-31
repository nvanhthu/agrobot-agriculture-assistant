# Agrobot – Agriculture RAG Chatbot
## Overview

Agrobot is a chatbot designed to answer questions about farming practices in British Columbia.

It uses a Retrieval-Augmented Generation (RAG) approach, where relevant information is first retrieved from agriculture documents and then passed to a language model to generate grounded answers.

The goal is to make responses more accurate and based on real documents instead of general model knowledge.

A simple Streamlit interface is used for interaction.

---

## Demo

![Agrobot Demo](images/demo.jpg)

---

## How it works

### 1. Text Extraction
**File:** `text_extraction.ipynb`

- Loads agriculture PDFs from the `Data/` folder  
- Extracts raw text  
- Cleans and formats it  
- Saves output into `Processed_Data/`

---

### 2. Embedding & Indexing
**File:** `embedding_and_indexing.py`

- Splits text into chunks (~1000 tokens)  
- Converts chunks into embeddings using OpenAI  
- Stores embeddings in a FAISS index  
- Reuses existing index if available  

---

### 3. Chatbot System

#### Backend Logic
**File:** `rag_utils.py`

- Loads processed documents  
- Splits into chunks  
- Creates embeddings  
- Uses FAISS for similarity search  
- Sends retrieved context + question to GPT-3.5-turbo  

If no relevant information is found, the system asks the user to rephrase.

---

#### Frontend
**File:** `interface.py`

- Built with Streamlit  
- Simple chat interface  
- Real-time responses  
- Maintains chat history using session state  

---

## Pipeline Flow

User Question → FAISS Search → Relevant Context → GPT-3.5-turbo → Answer

---

## Tech Stack

- Python  
- OpenAI API  
- FAISS  
- Streamlit  
- Jupyter Notebook  

---

## Key Features

- Semantic search over agriculture documents  
- Retrieval-Augmented Generation (RAG) pipeline  
- Context-aware answers grounded in real data  
- Fast similarity search with FAISS  
- Simple chat UI  

---

## What this project shows

- End-to-end RAG system  
- Practical use of embeddings + LLMs  
- Real-world agriculture Q&A assistant  
- Full pipeline: data → retrieval → generation → UI  

---

## Future Improvements

- Better retrieval accuracy  
- More agriculture datasets  
- Web deployment  
- Conversation memory  
- UI improvements  

---

## Author

Thu Nguyen 
RAG-based agriculture assistant project
