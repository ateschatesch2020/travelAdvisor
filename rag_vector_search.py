from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
load_dotenv()

embedding_model  = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"))

persist_directory = "./chroma_db"

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

def arama(question):
    print(f"Question is being searched: {question} ...")
    print("-"*50)

    results = vectorstore.similarity_search_with_score(question, k=2)

    if not results:
        print("There are no results")
        return
    
    for i, (document, puan) in enumerate(results):
        print(f"Content found: {i+1}")
        print(f"Content: {document.page_content}")
        print(f"Source: {document.metadata}")
        print(f"similarity score: {puan:.4f}")
        print("-"*20)

arama("ABD'de gunluk hotel konaklama fiyati limiti nedir?")



