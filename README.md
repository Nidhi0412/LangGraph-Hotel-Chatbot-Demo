# LangGraph Hotel Chatbot Demo — Q&A and Insights Agent

**Demo:** Multi-agent AI chatbot for hotel services — **Q&A**, **insights** (BI-style), and other intents, built with **LangGraph** and **FastAPI**.

---

## What this is

- **Q&A agent:** General hotel questions (policies, services, concierge).
- **Insights agent:** Category-wise business insights (revenue, booking pace, market/channel, operations, direct-channel) — optional, depends on backend data/API.
- **Other intents:** Check-in, check-out, offers, feedback, lost & found; each handled by a dedicated agent node.
- **Stack:** LangGraph (orchestration), LangChain/OpenAI (LLM), FastAPI (REST + WebSocket), simple HTML chat UI.

Good for **portfolio/demo**: shows agentic workflow, intent routing, and plug-in style for a real API (insights).

---

## Quick start

```bash
cd Langgraph_demo
pip install -r requirements.txt
```

Create a `.env` in this folder (see **Security** below):

```env
OPENAI_API_KEY=your_openai_key_here
```

Run the API and open the chat UI:

```bash
uvicorn fastapi_chatbot:app --reload --host 0.0.0.0 --port 8000
```

Open: **http://localhost:8000** — you get the chat interface.  
Use **http://localhost:8000/docs** for the OpenAPI docs.

---

## Project layout

| File / folder      | Purpose |
|--------------------|--------|
| `fastapi_chatbot.py` | FastAPI app: `/chat`, `/conversation/{guest_id}`, WebSocket, static chat UI. |
| `langgraph_new.py`   | LangGraph workflow: intent classification → agent nodes (QA, insights, check-in, etc.) → merge. |
| `Insights_new.py`    | Insights generation (Streamlit + async wrapper for LangGraph). **Uses S3/OpenAI — requires credentials; use env vars.** |
| `static/index.html`  | Simple chat UI. |
| `analysis.txt`       | Short doc: how to add a new API/agent into the LangGraph chatbot. |
| `requirements.txt`   | Python dependencies. |

---

## Using this in a Git repo / portfolio

- **Do not commit** real API keys or AWS credentials. Use a `.env` file (and add `.env` to `.gitignore`) and set `OPENAI_API_KEY` (and any S3 vars) in the environment.
- **Include:** `fastapi_chatbot.py`, `langgraph_new.py`, `static/`, `requirements.txt`, `README.md`, `analysis.txt`. Optionally a **stub** or mock for `Insights_new` (e.g. return placeholder text) if you want the repo to run without S3/OpenAI.
- **Description line** for the repo:  
  `Multi-agent hotel chatbot (Q&A + insights) with LangGraph and FastAPI — demo.`
- You can **link** this from your main GenAI portfolio README as: “LangGraph demo: Q&A and insights agent.”

---

## Security (important)

- **OpenAI:** Set `OPENAI_API_KEY` in `.env` or environment; remove any hardcoded key from the code before pushing.
- **Insights/S3:** If you use `Insights_new.py` with real data, use env vars for AWS keys and bucket/prefix; do not commit them.
- Add `.env` to `.gitignore` and commit a `.env.example` with placeholder variable names only.

---

## Tech stack

- **LangGraph** — workflow and agent routing  
- **LangChain / LangChain-OpenAI** — LLM calls and prompts  
- **FastAPI** — REST and WebSocket API  
- **Pydantic** — request/response models  

---

## Reference

- `analysis.txt` — checklist for integrating a new API (e.g. another insights source) into the LangGraph graph.

---

## Repository

**[github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo](https://github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo)**
