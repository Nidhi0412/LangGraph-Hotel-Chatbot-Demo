# Push to GitHub — LangGraph Hotel Chatbot Demo

**Repo:** [github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo](https://github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo)

Run these from the `Langgraph_demo` folder:

```bash
cd ~/Langchain_tutorials/Practice/GenAI_APP/Langgraph_demo
git init
git add .
git status
git commit -m "LangGraph hotel chatbot: Q&A + insights agents, FastAPI, env-based config"
git branch -M main
git remote add origin https://github.com/Nidhi0412/LangGraph-Hotel-Chatbot-Demo.git
git push -u origin main --force
```

Use a **Personal Access Token** when Git asks for a password (do not paste the token in chat).  
`--force` is only needed because the remote already has one commit (initial README); after this use normal `git push`.

**Ensure `.env` is not committed** — add a `.gitignore` in this folder with:
```
.env
__pycache__/
*.pyc
*.pyo
```
