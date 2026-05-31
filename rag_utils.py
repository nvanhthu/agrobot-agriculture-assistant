import os
import numpy as np
import faiss
from openai import OpenAI
from typing import List
import tiktoken


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536
MAX_TOKENS = 1000  # chunk size for splitting text
TOP_K = 2  # number of top chunks to retrieve

MAX_MODEL_TOKENS = 16385  # model token limit for GPT-3.5-turbo
RESERVED_TOKENS_FOR_ANSWER = 500  # tokens reserved for answer generation

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def load_text_files(folder_path: str) -> List[str]:
    """Load all .txt files from folder and return list of raw text strings."""
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                chunks.append(f.read())
    return chunks


def split_text(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into chunks approximately max_tokens tokens each."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += 1
        current_chunk.append(word)
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings from OpenAI API for a list of text chunks."""
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding_vector = np.array(response.data[0].embedding, dtype=np.float32)
        embeddings.append(embedding_vector)
    return embeddings


def save_faiss_index(embeddings: List[np.ndarray], file_path="Processed_Data/faiss_index.bin"):
    """Save FAISS index to disk."""
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(np.vstack(embeddings))
    faiss.write_index(index, file_path)


def load_faiss_index(file_path="Processed_Data/faiss_index.bin"):
    """Load FAISS index from disk."""
    if os.path.exists(file_path):
        return faiss.read_index(file_path)
    else:
        raise FileNotFoundError(f"FAISS index not found at {file_path}")


def retrieve_similar_chunks(query: str, texts: List[str], index, top_k=TOP_K) -> List[str]:
    """Retrieve top-k most similar text chunks to the query using FAISS."""
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    ).data[0].embedding
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [texts[i] for i in indices[0]]


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate answer using OpenAI chat completion with context chunks and question."""
    context = ""
    for chunk in context_chunks:
        chunk_tokens = len(tokenizer.encode(chunk))
        question_tokens = len(tokenizer.encode(question))
        temp_prompt = (
            "You are an assistant helping farmers in British Columbia. "
            "Based only on the information below, answer the user's question.\n\n"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        prompt_tokens = len(tokenizer.encode(temp_prompt))

        if prompt_tokens + chunk_tokens + RESERVED_TOKENS_FOR_ANSWER > MAX_MODEL_TOKENS:
            break
        context += "\n\n" + chunk

    prompt = (
        "You are an assistant helping farmers in British Columbia. "
        "Based only on the information below, answer the user's question.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
