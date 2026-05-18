import sqlite3
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api import app, chatbot
from rate_limiter import RateLimiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setattr(chatbot, "db_file_path", db)
    monkeypatch.setattr(chatbot, "connection_string", f"sqlite:///{db}")
    chatbot._init_session_db()
    # pre-create message_store so delete_session doesn't 500 on a fresh db
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS message_store "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, message TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()


# ── helpers ───────────────────────────────────────────────────────────────────

def _create(user_id="u1", title="Test"):
    return client.post("/sessions/create", json={"user_id": user_id, "title": title})


# ── POST /sessions/create ────────────────────────────────────────────────────

def test_create_session_returns_session_id():
    res = _create()
    assert res.status_code == 200
    assert "session_id" in res.json()


def test_create_session_rejects_blank_title():
    res = client.post("/sessions/create", json={"user_id": "u1", "title": "   "})
    assert res.status_code == 422


def test_create_session_rejects_empty_user_id():
    res = client.post("/sessions/create", json={"user_id": "", "title": "T"})
    assert res.status_code == 422


# ── GET /sessions/{user_id} ──────────────────────────────────────────────────

def test_list_sessions_empty():
    res = client.get("/sessions/u1")
    assert res.status_code == 200
    assert res.json()["sessions"] == []


def test_list_sessions_returns_created():
    sid = _create().json()["session_id"]
    sessions = client.get("/sessions/u1").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == sid


# ── PATCH /sessions/{session_id}/rename ──────────────────────────────────────

def test_rename_session():
    sid = _create(title="Old").json()["session_id"]
    res = client.patch(f"/sessions/{sid}/rename", json={"title": "New"})
    assert res.status_code == 200
    assert res.json()["title"] == "New"


# ── DELETE /sessions/{session_id} ────────────────────────────────────────────

def test_delete_session_removes_it():
    sid = _create().json()["session_id"]
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.get("/sessions/u1").json()["sessions"] == []


# ── GET /history/{session_id} ────────────────────────────────────────────────

def test_history_empty_for_new_session():
    sid = _create().json()["session_id"]
    res = client.get(f"/history/{sid}")
    assert res.status_code == 200
    assert res.json()["messages"] == []


# ── POST /chat ───────────────────────────────────────────────────────────────

def test_chat_streams_response():
    sid = _create().json()["session_id"]
    with patch.object(chatbot, "chat_stream", return_value=iter(["Hello", " world"])):
        res = client.post("/chat", json={"session_id": sid, "query": "Hi"})
    assert res.status_code == 200
    assert res.text == "Hello world"


def test_chat_rejects_blank_query():
    res = client.post("/chat", json={"session_id": "any", "query": "   "})
    assert res.status_code == 422


def test_chat_rejects_empty_session_id():
    res = client.post("/chat", json={"session_id": "", "query": "Hi"})
    assert res.status_code == 422


# ── GET /rate-status ─────────────────────────────────────────────────────────

def test_rate_status_returns_headers():
    fake = {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "95"}
    with patch.object(RateLimiter, "get_rate_limit_headers", return_value=fake):
        res = client.get("/rate-status")
    assert res.status_code == 200
    assert res.json() == fake


def test_rate_status_returns_empty_when_unavailable():
    with patch.object(RateLimiter, "get_rate_limit_headers", return_value={}):
        res = client.get("/rate-status")
    assert res.status_code == 200
    assert res.json() == {}
