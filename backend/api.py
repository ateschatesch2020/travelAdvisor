import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from chatbot import ChatbotManager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Chatbot Manager API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

chatbot = ChatbotManager()

class CreateSessionRequest(BaseModel):
    user_id: str
    title: str = "New Chat Session"

class ChatRequest(BaseModel):
    session_id: str
    query: str  

class MessageResponse(BaseModel):
    content: str
    role: str = "assistant"

class RenameSessionRequest(BaseModel):
    title: str

@app.post("/sessions/create")
def create_session(request: CreateSessionRequest):
    try:
        session_id = chatbot.create_session(user_id=request.user_id, title=request.title)
        return {"session_id": session_id}
    except Exception as e:
        logger.error("create_session failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        session_id = chatbot.delete_session(session_id=session_id)
        return {"session_id": session_id}
    except Exception as e:
        logger.error("delete_session failed for %s", session_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{user_id}")
def list_sessions(user_id: str):
    try:
        sessions = chatbot.list_sessions(user_id=user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error("list_sessions failed for user %s", user_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/sessions/{session_id}/rename")
def rename_session(session_id: str, request: RenameSessionRequest):
    try:
        chatbot.update_session_title(session_id=session_id, new_title=request.title)
        return {"session_id": session_id, "title": request.title}
    except Exception as e:
        logger.error("rename_session failed for %s", session_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
def get_history(session_id: str):
    try:
        messages = chatbot.get_messages(session_id=session_id)
        result = []
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            result.append(MessageResponse(content=msg.content, role=role))
        return {"messages": result}
    except Exception as e:
        logger.error("get_history failed for %s", session_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    def iterate_responses():
        try:
            for response in chatbot.chat_stream(session_id=request.session_id, query=request.query):
                yield response
        except Exception:
            logger.error("Unhandled error in chat stream for session %s", request.session_id, exc_info=True)
            yield "Sorry, I encountered an error while processing your request."

    return StreamingResponse(iterate_responses(), media_type="text/plain")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
    

