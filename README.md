# rlm-api

**Recursive Language Model (RLM) as an API** — DSPy RLM + Robyn + Neon Postgres memory layer.

## Architecture

```
Client
  │ POST /v1/rlm/runs
  ▼
Robyn API ── auth ──► asyncio task
                         │
                         ▼
                   Memory recall (Neon pgvector + FTS)
                         │
                         ▼
                   dspy.RLM execution
                   root LM: plan/explore
                   sub_lm: extract/classify
                         │
                         ▼
                   Memory write (run_summary)
                         │
  GET /v1/rlm/runs/:id ◄─┘
```

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/svngoku/rlm-api
cd rlm-api
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in DATABASE_URL, RLM_ROOT_MODEL, RLM_SUB_MODEL,
# EMBEDDING_MODEL, OPENROUTER_API_KEY, API_SECRET_KEY

# 3. Run migrations against Neon
psql $DATABASE_URL -f migrations/001_memory.sql
psql $DATABASE_URL -f migrations/002_search_fn.sql

# 4. Start
python app.py
```

## Docker

```bash
docker build -t rlm-api .
docker run -p 8080:8080 --env-file .env rlm-api
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/rlm/runs` | Submit a new RLM run |
| `GET` | `/v1/rlm/runs/:id` | Poll run status and result |
| `POST` | `/v1/memories` | Write a memory item |
| `GET` | `/v1/memories/search?tenant_id=&subject_id=&q=` | Hybrid memory search |
| `GET` | `/healthz` | Health check |

### POST /v1/rlm/runs

```json
{
  "tenant_id": "org-1",
  "subject_id": "user-42",
  "namespace": "default",
  "context": "<large document or repo content>",
  "query": "Find all breaking API changes and return migration steps.",
  "limits": {
    "max_iters": 12,
    "max_llm_calls": 24,
    "max_output_chars": 8000,
    "timeout_s": 300
  },
  "include_trajectory": false
}
```

### Authentication

Pass `Authorization: Bearer <API_SECRET_KEY>` on every request except `/healthz`.

## Memory kinds

| Kind | Purpose |
|------|---------|
| `fact` | Stable, verified facts about a repo, project, or user |
| `preference` | User or tenant preferences |
| `decision` | Approved design/architecture decisions |
| `episode` | Short-term episodic context |
| `run_summary` | Auto-written after each RLM run |
| `feedback` | Explicit user feedback on a result |

## Stack

- **[DSPy](https://dspy.ai)** — RLM module, programmatic LM orchestration
- **[Robyn](https://robyn.tech)** — Rust-runtime async Python API framework
- **[Neon](https://neon.tech)** — Serverless Postgres with pgvector
- **[asyncpg](https://github.com/MagicStack/asyncpg)** — High-performance async Postgres driver
- **[OpenRouter](https://openrouter.ai)** — Unified LLM API (root, sub, embedding models)
