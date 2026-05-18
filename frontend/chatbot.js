class ChatBot {
  constructor() {
    this.API_URL = "http://localhost:8001";
    this.USER_ID = "user123";
    this.currentSessionId = null;
    this.abortController = null;
  }

  async init() {
    const newSessionBtn = document.getElementById("newSessionBtn");
    const resetBtn = document.getElementById("resetBtn");
    const sendBtn = document.getElementById("sendBtn");
    const userInput = document.getElementById("userInput");
    const deleteSessionBtn = document.getElementById("deleteSessionBtn");

    if (newSessionBtn) newSessionBtn.addEventListener("click", () => this.createNewSession());
    if (resetBtn) resetBtn.addEventListener("click", () => this.resetSession());
    if (deleteSessionBtn) deleteSessionBtn.addEventListener("click", ()=> this.deleteSession())
    if (sendBtn) {
      sendBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.handleChatAction();
      });
    }
    
    // enter button implementation:
    if (userInput) {
      userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();//prevent new line
          this.sendMessage();
        }
      });
      userInput.focus();// userInput
    }

    const userDisplayName = document.getElementById("userDisplayName");
    if (userDisplayName) userDisplayName.textContent = this.USER_ID;

    const userDisplayCharacter = document.getElementById("userDisplayCharacter");
    if (userDisplayCharacter) userDisplayCharacter.textContent = this.USER_ID.slice(0, 2).toUpperCase();
    
    //load all sessions on the left side of page
    await this.loadSessions();

    //load chats of the active session
    const savedSession = localStorage.getItem("activeSession");
    // activeSession can be found on the browser F12 -> Application tab -> local storage
    if (savedSession) {
      await this.loadChatHistory(savedSession);
    }
  }

  handleChatAction() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
      return;
    }
    this.sendMessage();
  }

  resetSession() {
    localStorage.removeItem("activeSession");
    window.location.reload();
  }
  async renameSession(session_id, currentTitle) {
    const newTitle = prompt("New title:", currentTitle);
    if (!newTitle || newTitle.trim() === currentTitle) return;
    try {
      const res = await fetch(`${this.API_URL}/sessions/${session_id}/rename`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle.trim() })
      });
      if (res.ok) await this.loadSessions();
    } catch (e) {
      this.showToast("Could not rename session — is the server running?");
    }
  }

  async deleteSession(session_id) {
    try {
      const res = await fetch(`${this.API_URL}/sessions/${session_id}`, {
        method: "DELETE"
      });
      if (res.ok) {
        if (this.currentSessionId === session_id) {
          this.currentSessionId = null;
          localStorage.removeItem("activeSession");
          document.getElementById("chatContainer").innerHTML = `
            <div id="welcomeScreen" class="flex flex-col items-center justify-center h-full text-center animate-fade-in px-4">
              <div class="w-24 h-24 bg-white rounded-3xl flex items-center justify-center mb-6 shadow-xl shadow-gray-200 border border-gray-100">
                <i class="fa-solid fa-robot text-4xl text-blue-600"></i>
              </div>
              <h2 class="text-2xl font-bold text-gray-800 mb-2">Hello!</h2>
              <p class="text-gray-500 max-w-md">How can I help you?</p>
            </div>`;
        }
        await this.loadSessions();
      }
    } catch (e) {
      this.showToast("Could not delete session — is the server running?");
    }
  }
  async sendMessage() {
    const input = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const text = input.value.trim();

    if (!text) return;

    if (!this.currentSessionId) {
      alert("Please start a new chat first.");
      return;
    }

    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    input.value = "";
    sendBtn.innerHTML = "<i class='fa-solid fa-stop'></i>";
    sendBtn.classList.remove("bg-blue-600", "hover:bg-blue-700");
    sendBtn.classList.add("bg-red-600", "hover:bg-red-700");

    const chatContainer = document.getElementById("chatContainer");
    if (chatContainer.innerHTML.includes("Empty Chat")) chatContainer.innerHTML = "";

    // add user message
    this.appendMessage("user", text);
    this.scrollToBottom();

    // assistant is thinking
    const botBubble = this.appendMessage("assistant", "...");
    const contentDiv = botBubble.querySelector(".message-content");//get the element with .(class) message-content
    contentDiv.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Thinking...';

    try {
      const res = await fetch(`${this.API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: this.currentSessionId, query: text }),
        signal: signal
      });

      //getting the response of llm as stream
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let isFirstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (signal.aborted) break;

        if (isFirstChunk) {
          contentDiv.innerHTML = "";
          isFirstChunk = false;
        }

        const chunk = decoder.decode(value, { stream: true });
        contentDiv.textContent += chunk;
        this.scrollToBottom();
      }
    } catch (err) {
      if (err.name === "AbortError") {
        contentDiv.innerHTML = "<span class='text-xs text-red-500 italic'>Interrupted.</span>";
      } else if (err instanceof TypeError) {
        contentDiv.innerHTML = "<span class='text-red-500'>⚠️ Could not reach the server.</span>";
        this.showToast("Connection error — is the server running?");
      } else {
        contentDiv.innerHTML = "<span class='text-red-500'>⚠️ An error occurred.</span>";
        this.showToast("An unexpected error occurred.");
      }
    } finally {
      this.abortController = null;
      sendBtn.innerHTML = "<i class='fa-solid fa-paper-plane'></i>";
      sendBtn.classList.remove("bg-red-600", "hover:bg-red-700");
      sendBtn.classList.add("bg-blue-600", "hover:bg-blue-700");
      this.scrollToBottom();
      input.focus();
    }
  }

  async loadSessions() {
    try {
      const res = await fetch(`${this.API_URL}/sessions/${this.USER_ID}`);
      const { sessions } = await res.json();
      const listContainer = document.getElementById("sessionList");

      if (sessions.length === 0) {
        listContainer.innerHTML = '<div class="text-center text-gray-400 text-xs mt-4">No sessions created yet.</div>';
        return;
      }

      listContainer.innerHTML = "";

      sessions.forEach(session => {
        const btn = document.createElement("button");
        const isActive = session.session_id === this.currentSessionId;

        btn.className = `
          w-full text-left p-3 rounded-xl text-sm mb-1 transition-all flex items-center gap-3 border group ${isActive ?
          "bg-blue-50 text-blue-700 border-blue-200 font-medium" :
          "text-gray-600 hover:bg-gray-50 hover:text-gray-900 border-transparent"
        }`;

        btn.innerHTML = `
          <i class="fa-regular fa-message flex-shrink-0 ${isActive ? "text-blue-600" : "text-gray-400"}"></i>
          <span class="truncate flex-1">${session.title}</span>
          <span class="rename-btn flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center text-gray-400 hover:text-blue-500 hover:bg-blue-50 transition-all opacity-0 group-hover:opacity-100">
            <i class="fa-solid fa-pen text-xs"></i>
          </span>
          <span class="delete-btn flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50 transition-all opacity-0 group-hover:opacity-100">
            <i class="fa-solid fa-trash text-xs"></i>
          </span>
        `;

        btn.addEventListener("click", () => this.loadChatHistory(session.session_id));

        const renameBtn = btn.querySelector(".rename-btn");
        renameBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.renameSession(session.session_id, session.title);
        });

        const deleteBtn = btn.querySelector(".delete-btn");
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.deleteSession(session.session_id);
        });

        listContainer.appendChild(btn);
      });
    } catch (e) {
      this.showToast("Could not load sessions — is the server running?");
    }
  }

  async loadChatHistory(sessionId) {
    this.currentSessionId = sessionId;//assign the clicked session to currentsession
    localStorage.setItem("activeSession", sessionId);
    this.loadSessions();

    try {
      const res = await fetch(`${this.API_URL}/history/${sessionId}`);
      const { messages } = await res.json();
      const chatContainer = document.getElementById("chatContainer");

      chatContainer.innerHTML = "";

      if (messages.length === 0) {
        chatContainer.innerHTML = `<div class="flex flex-col items-center justify-center h-full animate-fade-in p-4">
          <div class="bg-amber-50 border border-amber-200 rounded-xl p-4 max-w-md w-full shadow-sm flex items-start gap-3">
            <div class="bg-amber-100 p-2 rounded-lg text-amber-600"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div><h3 class="font-bold text-amber-800 text-sm">Empty Chat</h3>
            <p class="text-amber-700 text-xs mt-1">Lets write first message!</p></div></div></div>`;
        return;
      }

      messages.forEach((msg) => this.appendMessage(msg.role, msg.content));
      this.scrollToBottom();
    } catch (e) {
      this.showToast("Could not load chat history — is the server running?");
    }
  }

  async createNewSession() {
    const title = prompt("Chat title:", "New Topic");
    if (!title) return;

    try {
      const res = await fetch(`${this.API_URL}/sessions/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: this.USER_ID, title: title }),
      });
      const data = await res.json();
      await this.loadChatHistory(data.session_id);
    } catch (error) {
      this.showToast("Could not create session — is the server running?");
    }
  }

  appendMessage(role, text) {
    const container = document.getElementById("chatContainer");
    const isUser = role === "user";
    const div = document.createElement("div");
    div.className = `flex gap-4 ${isUser ? "flex-row-reverse" : "flex-row"} animate-fade-in group w-full`;

    const avatar = isUser
      ? `<div class="w-9 h-9 rounded-full bg-blue-600 flex-shrink-0 flex items-center justify-center text-xs font-bold text-white shadow-sm ring-2 ring-white">U</div>`
      : `<div class="w-9 h-9 rounded-full bg-white flex-shrink-0 flex items-center justify-center text-xs text-blue-600 shadow-sm border border-gray-200"><i class="fa-solid fa-robot text-lg"></i></div>`;

    const bubbleStyle = isUser
      ? "bg-blue-600 text-white rounded-2xl rounded-tr-none shadow-md shadow-blue-100"
      : "bg-white text-gray-800 border border-gray-200 rounded-2xl rounded-tl-none shadow-sm";

    div.innerHTML =
      avatar +
      `<div class="max-w-[85%] md:max-w-[75%] min-w-0"><div class="message-content text-[15px] leading-relaxed py-3.5 px-5 break-text ${bubbleStyle}">${isUser ? this.escapeHtml(text) : text}</div></div>`;

    container.appendChild(div);
    return div;
  }

  escapeHtml(text) {
    if (!text) return text;
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  scrollToBottom() {
    const container = document.getElementById("chatContainer");
    if (container) {
      setTimeout(() => {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: "smooth",
        });
      }, 50);
    }
  }

  showToast(message) {
    const toast = document.createElement("div");
    toast.className =
      "fixed top-4 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-xl shadow-lg text-sm font-medium bg-red-50 text-red-700 border border-red-200 animate-fade-in";
    toast.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i><span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  }
}
// to ensure that init method will be called after all elements were loaded.
document.addEventListener("DOMContentLoaded", () => {
  const chatbot = new ChatBot();
  chatbot.init();
});
