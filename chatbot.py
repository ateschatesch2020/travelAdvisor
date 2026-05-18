import os
import uuid
import sqlite3
import tools
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
from rate_limiter import RateLimiter
load_dotenv()


class ChatbotManager:
    def __init__(self, model_name: str = "openai/gpt-5.4-mini"):
        """ starts the chatbot
        """
        self.model_name = model_name
        self.db_file_path = "test_history.db"
        self.connection_string = f"sqlite:///{self.db_file_path}"
        self.model = ChatOpenRouter(
            model=self.model_name)

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

        For specific prices of flights use search_flights tool.
        For specific prices of hotels use search_hotels tool.
        For weather forecast for a specific date use search_weather tool.

        context:
        {context}
        """
        self.tools = tools.Tools.tools

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

    def _get_session_history(self, session_id: str):
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

        model_with_tools = self.model.bind_tools(self.tools)
        tool_map = {t.name: t for t in self.tools}

        def run_with_tools(inputs):
            prompt_messages = prompt.invoke(inputs).to_messages()

            # invoke() gives a proper AIMessage with reliably populated tool_calls
            ai_msg = model_with_tools.invoke(prompt_messages)

            if not ai_msg.tool_calls:
                # Stream the response token-by-token
                for chunk in model_with_tools.stream(prompt_messages):
                    yield chunk.content or ""
                return

            # Execute tools
            prompt_messages.append(ai_msg)
            for tc in ai_msg.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                prompt_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

            # Stream final response after tool results
            for chunk in model_with_tools.stream(prompt_messages):
                yield chunk.content or ""

        chain = (
            RunnablePassthrough.assign(
                search_query=rephrase_chain
            )
            |
            RunnablePassthrough.assign(
                context=itemgetter("search_query")
                | RunnableLambda(inspect_query)
                | self.retriever | self._format_docs,
            )
        | RunnableLambda(run_with_tools)
        )

        self.rephrase_chain = rephrase_chain
        self.run_with_tools = run_with_tools

        return RunnableWithMessageHistory(
            chain,
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
            print(f"An error occurred: {e}")
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
            print(f"An error occurred: {e}")
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
        config = {"configurable": {"session_id": session_id}}
        try:
            response = self.conversation_chain.invoke(
                {"question": query},
                config=config)
            return response
        except openai.RateLimitError:
            return "Rate limit reached. Please try again in a moment."
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I encountered an error while processing your request."

    def chat_stream(self, session_id: str, query: str):
        history = self._get_session_history(session_id)
        try:
            messages = history.messages

            search_query = self.rephrase_chain.invoke(
                {"question": query, "history": messages}
            )
            context = self._format_docs(self.retriever.invoke(search_query))

            inputs = {
                "question": query,
                "history": messages,
                "context": context,
                "search_query": search_query,
            }

            accumulated = ""
            for token in self.run_with_tools(inputs):
                accumulated += token
                yield token

            history.add_user_message(query)
            history.add_ai_message(accumulated)

        except openai.RateLimitError:
            yield "Rate limit reached. Please try again in a moment."
        except Exception as e:
            import traceback
            print(f"Error occurred: {e}")
            traceback.print_exc()
            yield "Sorry, I encountered an error while processing your request."


if __name__ == "__main__":
    manager = ChatbotManager()
    user = "user123"
    session_id = manager.create_session(user, "Test Chat")
    print(manager.chat(session_id,
          "What are the direct flights from Munich to Madrid on 2026-06-10?"))
