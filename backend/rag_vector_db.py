from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

file_name = "./documents/Corporate_Travel_and_Expense_Policy.pdf"

loader = PyPDFLoader(file_name)
docs = loader.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)

document_pieces = spliter.split_documents(docs)

# embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
# Use OpenAI-compatible Embeddings via OpenRouter
embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"))

persist_directory = "./chroma_db"

print(f"creating vector database...")

vectorstore = Chroma.from_documents(
    documents=document_pieces,
    embedding=embedding_model,
    persist_directory=persist_directory
)

print(f"Successfull! Data is stored to {persist_directory}.")
