# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

* State your assumptions explicitly. If uncertain, ask.
* If multiple interpretations exist, present them - don't pick silently.
* If a simpler approach exists, say so. Push back when warranted.
* If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

* No features beyond what was asked.
* No abstractions for single-use code.
* No "flexibility" or "configurability" that wasn't requested.
* No error handling for impossible scenarios.
* If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

* Don't "improve" adjacent code, comments, or formatting.
* Don't refactor things that aren't broken.
* Match existing style, even if you'd do it differently.
* If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

* Remove imports/variables/functions that YOUR changes made unused.
* Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

* "Add validation" → "Write tests for invalid inputs, then make them pass"
* "Fix the bug" → "Write a test that reproduces it, then make it pass"
* "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Travel Advisory Chatbot

### Running the project

**Backend** (FastAPI, port 8001) — run from project root:
```bash
python backend/api.py
```

**Frontend** (Vite, port 5173):
```bash
cd frontend && npm run dev
```

**Populate vector DB** (run once, or when the source PDF changes) — run from project root:
```bash
python backend/rag_vector_db.py
```
Source PDF must be at `./documents/Corporate_Travel_and_Expense_Policy.pdf`.

**Run tests** — run from project root:
```bash
pytest backend/tests/
```

### Required environment variables (.env)
- `OPENROUTER_API_KEY` — LLM (gpt-4o-mini) and embeddings (text-embedding-3-small) via OpenRouter
- `SERPAPI_KEY` — Google Flights and Google Hotels via SerpAPI

### Security

- **Never pass API keys as URL query parameters.** Keys in URLs appear in logs, browser history, and proxy/referrer headers. Always pass them in HTTP request headers (e.g., `Authorization: Bearer <token>` or a service-specific header like `key: <token>`).
- **Never call external APIs with secret keys from the frontend.** All calls to SerpAPI, WeatherAPI, OpenRouter, or any keyed external service must go through the backend. The frontend only calls `localhost:8001`.

### Architecture

```
Browser (frontend/index.html + chatbot.js)
    │ HTTP streaming  POST /chat
    ▼
backend/api.py  (FastAPI)  → calls chatbot.chat_stream(session_id, query)
    ▼
backend/chatbot.py  ChatbotManager
    ├── RAG: Chroma vector DB (chroma_db/) ← populated from PDF by backend/rag_vector_db.py
    │         top-2 chunks injected as context into every message
    ├── Agent: LangGraph ReAct agent (myagent)
    │         tools defined in tools.py → search_flights, search_hotels (SerpAPI)
    │         memory: MemorySaver keyed by thread_id = session_id
    └── Session history: SQLite (test_history.db) via SQLChatMessageHistory
```

**Request flow:**
1. `chat_stream` retrieves k=2 RAG docs, prepends as context to the query
2. Calls `myagent.stream(stream_mode="messages")`
3. Filters `AIMessageChunk` with non-empty content → yields text tokens to API
4. API returns `StreamingResponse(text/plain)` to frontend
5. Frontend reads chunks via `fetch` + `ReadableStream`

### Adding a new tool

1. Define `@tool` function in `backend/tools.py` with a clear docstring (the LLM uses this to decide when to call it).
2. Add it to `Tools.tools = [...]`.
3. Restart `backend/api.py` — no prompt changes needed; the agent auto-discovers tools via function calling.

### Key design decisions
- `chat` / `chat_stream` use the agent (not `conversation_chain`); `_create_chain` / `conversation_chain` exist but are unused.
- `agent_prompt` intentionally does NOT hardcode tool names — tool docstrings handle routing.
- `thread_id` for agent memory = `session_id` from API requests.
