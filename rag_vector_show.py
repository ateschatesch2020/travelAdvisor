from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()


# embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
# Use OpenAI-compatible Embeddings via OpenRouter
embedding_model  = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"))

persist_directory = "./chroma_db"

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

results = vectorstore.get(include=["embeddings","documents","metadatas"])

total_chunks = len(results["ids"])
print(f"There are {total_chunks} chunks stored in database.\n")

first_string = results["documents"][0]
print(first_string)
print("-"*50)

first_vector = results["embeddings"][0]
print(first_vector)


