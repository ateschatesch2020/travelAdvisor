# Travel Advisory Chatbot

AI-powered travel chatbot with real-time flight search, hotel availability, weather forecasts,
and multi-city itinerary optimization — grounded in company travel policy via RAG.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env` and fill in:

```
OPENROUTER_API_KEY=...   # LLM (gpt-4o-mini) and embeddings via OpenRouter
SERPAPI_KEY=...           # Google Flights and Hotels via SerpAPI
```

## Running

All commands from the project root.

**Backend** (FastAPI, port 8001):

```bash
python backend/api.py
```

**Frontend** (Vite, port 5173):

```bash
cd frontend && npm run dev
```

**Populate vector DB** (once, or whenever the source PDF changes):

```bash
python backend/rag_vector_db.py
```

Source PDF must be at `./documents/Corporate_Travel_and_Expense_Policy.pdf`.

## Tests

```bash
pytest backend/tests/
```

## Architecture

```
Browser (frontend/index.html + chatbot.js)
    │ HTTP streaming  POST /chat
    ▼
backend/api.py  (FastAPI)  → chatbot.chat_stream(session_id, query)
    ▼
backend/chatbot.py  ChatbotManager
    ├── RAG: Chroma vector DB (chroma_db/) ← built by backend/rag_vector_db.py
    ├── Agent: LangGraph ReAct agent — tools in backend/tools.py
    └── Session history: SQLite (test_history.db)
```

| File | Purpose |
|---|---|
| `backend/api.py` | FastAPI app — POST /chat, /sessions, /history |
| `backend/chatbot.py` | ChatbotManager, LangGraph agent, Chroma retriever |
| `backend/tools.py` | LangChain tools: search_flights, search_hotels, search_weather, optimize_itinerary |
| `backend/itinerary_optimizer.py` | CP-SAT solver (OR-Tools) for multi-city trip optimization |
| `backend/rag_vector_db.py` | One-time script to build the Chroma vector DB from the PDF |
