---
title: Ask Harshil
emoji: 💬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: RAG assistant that answers questions about Harshil Patel
---

# Ask my AI: a RAG assistant for my portfolio

**🌐 Live on my portfolio: [harshilmayurpatel.github.io](https://harshilmayurpatel.github.io/)** (use the **Ask my AI** button on any page)

A small retrieval-augmented generation (RAG) service that answers recruiters' questions about me, my projects, skills, and experience. It powers the **Ask my AI** chat on my portfolio.

```
Visitor question
   │
   ▼
FastAPI /chat ──► FastEmbed (bge-small, local ONNX) ──► FAISS top-4 chunks
   │                                                        │
   │◄──────── Llama 3.3 70B via Groq, grounded in chunks ◄──┘
   ▼
Streamed answer (Server-Sent Events) + source sections
```

## Stack
| Part | Tool | Why |
|---|---|---|
| Orchestration | LangChain | Loaders, splitters, vector store and chat model in one interface |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` | Runs locally on CPU (ONNX), free, no API key |
| Vector search | FAISS | Fast in-memory similarity search, index built into the image |
| LLM | Llama 3.3 70B on Groq | Free tier, very fast streaming |
| API | FastAPI + SSE | Word-by-word streaming to the browser |
| Hosting | Docker on Hugging Face Spaces | Free container hosting |

## How it stays accurate and safe
- **Grounded answers:** the model may only use retrieved facts and says "I don't know" otherwise.
- **Curated knowledge base:** clean Markdown in [`knowledge/`](knowledge/), one chunk per section. Placeholder (`TODO`) sections are skipped.
- **Prompt-injection guard:** the visitor's message is treated as a question, never as instructions.
- **Abuse limits:** CORS restricted to the portfolio, per-IP rate limits (8/min, 60/day), length caps.
- **Graceful fallback:** if the LLM is unavailable, it returns the best-matching facts verbatim.
- **Accuracy test:** [`eval/`](eval/) checks retrieval (right section in the top 4) and answers (expected facts present).

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # add your free Groq key (console.groq.com)
python ingest.py                # build the FAISS index
uvicorn app.main:app --reload --port 7860
```
Open http://localhost:7860/docs to try `/chat`.

## Test
```bash
python -m eval.run_eval                          # in-process
python -m eval.run_eval --url https://<space>.hf.space   # deployed
```

## Deploy (Hugging Face Spaces)
1. Create a Space with the **Docker** SDK.
2. Push this repo to the Space.
3. In **Settings → Variables and secrets**, add the secret `GROQ_API_KEY`.

## API
`POST /chat` with `{"question": "...", "history": [{"role": "user", "content": "..."}]}` streams
`data: {"t": "..."}` events, then `data: {"done": true, "sources": [...]}`.

`POST /intent` with `{"text": "can you make it darker"}` returns `{"action": "dark", "via": "llm"}`. The portfolio's voice mic uses it to understand free-form sentences. The model can only pick from a fixed list of site actions (validated on the server), plus `ask` for questions and `none`. Without the LLM it falls back to local embedding similarity over example phrases.

`GET /health` reports status.
