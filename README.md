# rlm-api

A production-oriented API for DSPy Recursive Language Model (RLM) runs, with a
Robyn HTTP process, a durable PostgreSQL queue, and pgvector/full-text memory.

## Architecture

`POST /v1/rlm/runs` only writes a queued row. A separate `python worker.py`
process atomically claims work with `FOR UPDATE SKIP LOCKED`, recalls tenant
memory, executes `dspy.RLM`, writes a summary, and persists the result and
lifecycle events. API restarts therefore do not lose accepted work.

Workers heartbeat their active lease. Stale `running` rows are recovered and
retried with bounded exponential delays of 1, 2, 4, ... up to 300 seconds.
Attempts are capped per run (default 3, API maximum 10). Completion updates
require the same worker lease, preventing a stale worker from overwriting a
recovered run.

For a small single-container deployment, set `EMBEDDED_WORKER=true`. Production
deployments should independently scale:

```bash
python app.py       # API
python worker.py    # one or more workers
```

Graceful shutdown stops polling, lets the current embedded run complete, and
closes PostgreSQL pools.

## Configuration

Copy `.env.example` to `.env` and replace every `REPLACE_WITH_...` value.
Required settings are:

- `DATABASE_URL`
- `API_KEYS_JSON`
- `RLM_ROOT_MODEL`
- `RLM_SUB_MODEL`
- `EMBEDDING_MODEL`
- positive `EMBEDDING_DIM`

Model IDs are deliberately opaque environment values. Any current ID supported
by the deployed DSPy/LiteLLM provider can be selected without changing code.
This repository does not label an unverified model slug as "latest". Root and
sub IDs may match, but startup emits a warning because that can remove the
intended specialization.

The embedding provider must return exactly `EMBEDDING_DIM` values. Every result
is checked before SQL is executed, and startup verifies that
`memory_items.embedding` is `vector(EMBEDDING_DIM)`. Migrations 001/002 use
1536; choosing another dimension requires updating/reapplying the vector column
and search function schema.

`GET /v1/models` (authenticated) safely reports active IDs and dimension; it
never reports provider credentials.

## Database setup

Apply migrations in order:

```bash
psql "$DATABASE_URL" -f migrations/001_memory.sql
psql "$DATABASE_URL" -f migrations/002_search_fn.sql
psql "$DATABASE_URL" -f migrations/003_expire_cleanup.sql
psql "$DATABASE_URL" -f migrations/004_durable_run_queue.sql
```

Migration 003 is periodic cleanup SQL, so schedule it with your database or
operations scheduler. Migration 004 is idempotent and adds durable queue,
retry, heartbeat, context, trajectory, timestamp, and indexing fields.

Tenant isolation is enforced in the application: authenticated tenant identity
is included in every run and memory lookup. Migration 004 explicitly disables
the policy-free RLS flags left by migration 001; those flags did not constitute
working row policies. The database connection string is a server-only secret.
Never expose the database role or direct database access to API clients.

## Authentication and tenant semantics

`API_KEYS_JSON` maps opaque bearer keys to tenant principals:

```json
{
  "long-random-key-one": "tenant-a",
  "long-random-key-two": {
    "tenant_id": "tenant-b",
    "subject_id": "optional-fixed-subject",
    "role": "client"
  }
}
```

Send `Authorization: Bearer <key>`. Missing auth configuration fails startup,
and token comparisons are constant-time. Tenant identity comes from the key.
The optional legacy `tenant_id` request/query field is accepted only when it
matches that identity; mismatches return 403. A credential with `subject_id`
can only access that subject. Run reads and memory operations are tenant scoped.

`/livez` and `/healthz` are intentionally unauthenticated.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/rlm/runs` | Durably enqueue a run |
| `GET` | `/v1/rlm/runs/:run_id` | Read a tenant-scoped run |
| `POST` | `/v1/memories` | Write a tenant-scoped memory |
| `GET` | `/v1/memories/search` | Search tenant/subject memory |
| `GET` | `/v1/models` | Safe active model configuration |
| `GET` | `/livez` | Process liveness |
| `GET` | `/healthz` | PostgreSQL readiness (503 if unavailable) |

All responses include `X-Request-ID`; a valid-sized incoming value is preserved.
Logs are structured JSON and include request, run, and tenant context where
available. Internal exceptions are logged but not returned to clients.

### Enqueue example

```bash
curl -i http://localhost:8080/v1/rlm/runs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "user-42",
    "namespace": "default",
    "context": "The corpus to analyze",
    "query": "Answer using corpus evidence",
    "include_trajectory": false,
    "max_attempts": 3
  }'
```

The response is `202` with a `Location` header. Poll that URL with the same
tenant credential.

Memory `kind` is validated as one of `fact`, `preference`, `decision`,
`episode`, `run_summary`, or `feedback`.

## Local quality checks

Python 3.12 is required.

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

CI runs those checks on Python 3.12. Unit tests use fakes and require neither a
database nor a model provider.

## Docker

The image runs as an unprivileged user and checks `/healthz`.

```bash
docker build -t rlm-api .
docker run --env-file .env -p 8080:8080 rlm-api
docker run --env-file .env rlm-api python worker.py
```

Run the API and worker as separate services against the same database. The
default command is the API.

## Deployment checklist

1. Select and verify current provider model IDs and credentials.
2. Ensure embedding output and database vector dimensions match.
3. Generate long random API keys and map each to exactly one tenant.
4. Keep `DATABASE_URL` and provider/API keys server-only.
5. Apply migrations 001 through 004 and schedule migration 003 cleanup.
6. Deploy at least one API and one worker (or explicitly enable embedded mode).
7. Configure termination grace longer than the maximum run timeout.
8. Alert on `/healthz`, exhausted runs, stale recovery, and retry/failure logs.
9. Back up PostgreSQL according to the chosen provider's recovery policy.

Basic latency, attempt, and active root/sub-model metadata are stored in
`rlm_runs.usage`; state changes and retry metadata are stored in `rlm_events`.
