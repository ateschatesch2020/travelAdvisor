import logging
import os
import numpy as np
import json
import uuid
import sqlite3
import faiss
import tools
from datetime import date

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


class ChatbotManager:
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        """ starts the chatbot
        """
        self.model_name = model_name
        self.db_file_path = "test_history.db"
        self.connection_string = f"sqlite:///{self.db_file_path}"
        self.model = ChatOpenRouter(
            model=self.model_name)

        # self.embedding_model = HuggingFaceEmbeddings(
        #   model="sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"))
        persist_directory = "./chroma_db"
        self.vectorestore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model)
        self.retriever = self.vectorestore.as_retriever(search_kwargs={"k": 2})
        self._init_session_db()

        self.checkpointer = MemorySaver()

        self.rephrase_system = """
        Consider the history of the conversation. If the user uses pronouns like "it", "them",
        which cites the term in the conversation history, change the user input including the term instead of 
        that pronoun. If there aren't any pronouns of citation, don't change the query.
         
        Don't give answer, just give the corrected question.
        """

        self.system_prompt = """
        You are a travel advisor based on the regulations of the company.   
        If the answer is not in this context, say kindly that you don't know. Never make it up.
        When a question required to be searched or calculated, use the tools.
        For general advices or questions respond directly without calling tools.
        If the question is in turkish, answer in turkish, if the question is in german, answer in german etc.
        Use search_flights when you asked about the flights.

        context:
        {context}
        """
        self.agent_prompt = """
        You are a travel advisor with access to real-time tools.
        NEVER say you cannot find flight, hotel, or weather information — always use the available tools to look it up.
        Do not rely on training data for flight schedules, hotel availability, prices, or any real-time data; always call the appropriate tool.
        For hotel searches, use the hotel tool with the location name (city or area) as the parameter.
        For company policy questions, use the provided context.
        Reply in the same language as the user.
        When presenting hotel results, show each hotel image using markdown image syntax ![Hotel Name](image_url) at the start of each hotel entry.
        """

        self.tools = tools.Tools.tools
        self.myagent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.agent_prompt,
            checkpointer=self.checkpointer
        )

        # create chain
        self.conversation_chain = self._create_chain()

    def _init_session_db(self):
        """creates chat_sessions table in sqlite db"""

        conn = sqlite3.connect(self.db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """ gets the chat history of given session_id from sqlite database"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.connection_string
        )

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def _create_chain(self):
        """ creates the chain which combine the prompt and llm"""

        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", self.rephrase_system),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        rephrase_chain = rephrase_prompt | self.model | StrOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        def inspect_query(query):
            print(f"the question gone to retriever: {query} ")
            return query

        chain = (
            RunnablePassthrough.assign(
                search_query=rephrase_chain
                # rephrase_chain is executed here and the output is assigned to search_query
            )
            |
            RunnablePassthrough.assign(
                # the output of the previous item of the chain is given to RunnableLambda
                # to be written in inspect_query and then as input to retriever and then to format_docs
                context=itemgetter("search_query")
                | RunnableLambda(inspect_query)
                | self.retriever | self._format_docs,
            )
            # prompt needs context which is the previous item of the chain
            | prompt
            | self.model
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            chain,  # the response of the whole chain after StrOutputParser
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history")

    def create_session(self, user_id: str, title: str) -> str:
        """ creates a row in chat_sessions table."""
        session_id = str(uuid.uuid4())
        try:
            conn = sqlite3.connect(self.db_file_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_sessions (session_id, user_id, title)
                VALUES (?, ?, ?)
            ''', (session_id, user_id, title))
            conn.commit()
            conn.close()
            return session_id
        except Exception as e:
            logger.error("create_session failed for user %s", user_id, exc_info=True)
            return "Sorry, I encountered an error while processing your request."

    def delete_session(self, session_id: str) -> str:
        """Deletes session and its messages atomically."""
        conn = sqlite3.connect(self.db_file_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM message_store WHERE session_id = ?', (session_id,))
            cursor.execute(
                'DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
            conn.commit()
            return session_id
        except Exception as e:
            conn.rollback()
            logger.error("delete_session failed for %s", session_id, exc_info=True)
            raise
        finally:
            conn.close()

    def list_sessions(self, user_id: str):

        conn = sqlite3.connect(self.db_file_path)
        conn.row_factory = sqlite3.Row  # enable dict-like access to rows
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, title, created_at FROM chat_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        sessions = cursor.fetchall()
        conn.close()
        return [dict(session) for session in sessions]

    def update_session_title(self, session_id: str, new_title: str):
        conn = sqlite3.connect(self.db_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE chat_sessions
            SET title = ?
            WHERE session_id = ?
        ''', (new_title, session_id))
        conn.commit()
        conn.close()

    def get_messages(self, session_id: str):
        history = self._get_session_history(session_id)
        return history.messages

    def chat(self, session_id: str, query: str):
        """ end point method for chatting """
        response = "Sorry, I encountered an error while processing your request."
        try:
            docs = self.retriever.invoke(query)
            context = self._format_docs(docs)
            today = date.today().strftime("%Y-%m-%d")
            message = (f"Today's date: {today}.\n\nContext:\n{context}\n\nQuestion: {query}"
                       if context else f"Today's date: {today}.\n\nQuestion: {query}")
            result = self.myagent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": session_id}}
            )
            response = result["messages"][-1].content
        except Exception as e:
            logger.error("chat failed for session %s", session_id, exc_info=True)
        finally:
            history = self._get_session_history(session_id)
            history.add_user_message(query)
            history.add_ai_message(response)
        return response

    def chat_stream(self, session_id: str, query: str):
        full_response = "Sorry, I encountered an error while processing your request."
        try:
            docs = self.retriever.invoke(query)
            context = self._format_docs(docs)
            today = date.today().strftime("%Y-%m-%d")
            message = (f"Today's date: {today}.\n\nContext:\n{context}\n\nQuestion: {query}"
                       if context else f"Today's date: {today}.\n\nQuestion: {query}")
            full_response = ""
            for msg_chunk, metadata in self.myagent.stream(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode="messages"
            ):
                if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                    full_response += msg_chunk.content
                    yield msg_chunk.content
        except Exception as e:
            logger.error("chat_stream failed for session %s", session_id, exc_info=True)
            full_response = "Sorry, I encountered an error while processing your request."
            yield full_response
        finally:
            history = self._get_session_history(session_id)
            history.add_user_message(query)
            history.add_ai_message(full_response)

    def embed(self, text):
        response = self.embedding_model.embed_query(text)
        k = 2
        index = faiss.IndexFlatL2(1536)
        distances, indices = index.search(
            np.array([response]), k
        )
        return response

    def chat_by_vector(self, session_id: str, query: str) -> str:
        config = {"configurable": {"session_id": session_id}}
        try:
            vector = self.embed(query)
            docs = self.vectorestore.similarity_search_by_vector(vector, k=2)
            response = self.conversation_chain.invoke(
                {"question": self._format_docs(docs)},
                config=config)
            return response
        except Exception as e:
            logger.error("chat_by_vector failed for session %s", session_id, exc_info=True)
            return "Sorry, I encountered an error while processing your request."

    def chat_by_vector_stream(self, session_id: str, query: str):
        config = {"configurable": {"session_id": session_id}}
        try:
            vector = self.embed(query)
            docs = self.vectorestore.similarity_search_by_vector(vector, k=2)
            for doc in docs:
                yield doc.page_content
        except Exception as e:
            logger.error("chat_by_vector_stream failed for session %s", session_id, exc_info=True)
            yield "Sorry, I encountered an error while processing your request."

    def invoke_with_user(self, user_id: str, question: str):
        thread_id = f"user_session_{user_id}"
        return self.myagent.invoke({
            "messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}}
        )


if __name__ == "__main__":
    manager = ChatbotManager()

    user = "user123"

    session_id = manager.create_session(user, "Test Tools")

    # print(manager.chat(session_id,
    #        "What is the hotel price limit in USA?"))

    # print(manager.chat(session_id,
    #         "What about south america?"))

    response1 = manager.invoke_with_user(
        user, "I want to you list me the flights on 23.05.2026 from Munich to Madrid direct only, all airlines")
    print("1.Response: ", response1["messages"][-1].content)

    response2 = manager.invoke_with_user(
        user, "I want to see the return flights between 30.05 and 03.06")
    print("2.Response: ", response2["messages"][-1].content)
    # print(manager.chat_by_vector(session_id,
    #        "Temel gelir desteğinin faydaları özellikle hangi alanlara yönelik olmalıdır?"))

    # print(manager.chat_by_vector(session_id,
    #        "Buna ücretli iş de dahil mi?"))
    # session_id = manager.create_session("Ates Ates", "Turkish Search Test")
    # turkish_question = "Temel gelir desteği yardımlarının ana amaçları nelerdir?"
    # print(manager.chat_by_vector(session_id, turkish_question))
