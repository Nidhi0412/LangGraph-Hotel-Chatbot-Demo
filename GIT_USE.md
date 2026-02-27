# Using Langgraph_demo in Your Git Projects

## Option 1: Link from another repo (e.g. portfolio README)

Add a line like:

```markdown
- **[LangGraph Hotel Chatbot Demo](path/to/Langgraph_demo)** — Multi-agent Q&A and insights agent (LangGraph + FastAPI).
```

Or if this folder lives in a different repo:

```markdown
- **LangGraph demo:** [github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo](https://github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo) — Q&A + insights agent.
```

---

## Option 2: GitHub repo (standalone demo)

**Repo:** [github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo](https://github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo)

1. ~~Create a new repo~~ — Done.
2. **Secrets:** Code now uses `os.getenv()` for OpenAI and AWS; no hardcoded keys. Use `.env` locally (see `.env.example`).
3. **Push:** From `Langgraph_demo` folder run the commands in `PUSH_STEPS.md`.

**Repo description suggestion:**  
`Multi-agent hotel chatbot (Q&A + insights) with LangGraph and FastAPI — demo.`

---

## Option 3: Keep inside GenAI_APP repo

If your main repo is `GenAI_APP`, keep `Langgraph_demo/` as a subfolder and mention it in the root README (e.g. “LangGraph demo: Q&A and insights agent under `Langgraph_demo/`”). Ensure `.env` is in `.gitignore` and no secrets are committed.

---

## One-line summary for CV / profile

**LangGraph + FastAPI demo:** Multi-agent hotel chatbot (Q&A and BI-style insights) with intent classification and pluggable agent nodes.
