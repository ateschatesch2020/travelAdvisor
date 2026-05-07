from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()


embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"))

persist_directory = "./chroma_db"

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

# sonuclar = vectorstore.similarity_search_with_score("Filtresini nasıl temizlerim?", k=2)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

chat_model = ChatOpenRouter(model="gpt-4o-mini", temperature=0)

template = """
    you are a travel advisor based on the regulations of the company.
    answer the questions from given context.
    If the answer is not in this context, say kindly that you don't know. Never make it up.
    Answer not to exceed 3 sentences and the same language with the question. 
    If the question is in turkish, answer in turkish, if the question is in german, answer in german etc.

    {context}

    Question: {question}
"""


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])# without metadatas


prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

sonuc = chain.invoke(
    "ABD'de gunluk hotel konaklama fiyati limiti nedir?")

print(sonuc)

