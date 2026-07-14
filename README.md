# rlm-api

A production-oriented API for DSPy Recursive Language Model (RLM) runs, with a
Robyn HTTP process, a durable PostgreSQL queue, and pgvector/full-text memory.

## Architecture

`POST /v1/rlm/runs` only writes a queued row. A separate `python worker.py`
process atomically claims work with `FOR UPDATE SKIP LOCKED`, recalls tenant
memory asynchronously, and executes `dspy.RLM` in a fresh spawn-based child
process. The parent event loop remains responsive to timeouts and heartbeats and
terminates the child on cancellation or lease loss. Success and a durable
summary outbox row are persisted in one transaction. Workers independently
deliver idempotent summary side effects from that outbox. Lifecycle events are
persisted throughout, so API or worker restarts do not lose accepted work.

Workers heartbeat their active lease. Stale `running` rows are recovered and
retried with bounded exponential delays of 1, 2, 4, ... up to 300 seconds.
Attempts are capped per run (default 3, API maximum 10). Completion updates
require the same worker lease, preventing a stale worker from overwriting a
recovered run. One transient heartbeat exception does not imply lease loss;
continuous uncertainty near the stale deadline cancels the child defensively.

For a small single-container deployment, set `EMBEDDED_WORKER=true`. Production
deployments should independently scale:

```bash
python app.py       # API
python worker.py    # one or more workers
```

Graceful shutdown stops polling, cancels the active run and terminates its child
process, then closes PostgreSQL pools. The stale-run recovery path safely
requeues that interrupted lease.

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

Worker operations can tune `WORKER_BATCH_SIZE`, `WORKER_POLL_SECONDS`,
`WORKER_STALE_SECONDS`, `WORKER_RECOVERY_BATCH_SIZE` (default 100), and
`WORKER_MAX_POLL_FAILURES` (default 5).

## Database setup

Apply migrations in order:

```bash
psql -v ON_ERROR_STOP=1 "$DATABASE_URL" -f migrations/001_memory.sql
psql -v ON_ERROR_STOP=1 "$DATABASE_URL" -f migrations/002_search_fn.sql
psql -v ON_ERROR_STOP=1 "$DATABASE_URL" -f migrations/004_durable_run_queue.sql
```

Run migration 004 with psql autocommit and do not wrap it in an explicit
transaction; its operational indexes are built concurrently.

Migration 003 is periodic cleanup SQL, not a schema migration. Schedule it with
your database or operations scheduler. Migration 004 is idempotent and adds
durable queue, retry, heartbeat, context, trajectory, timestamp, indexing, and
idempotent run-summary outbox fields. Unrecoverable legacy queued/running rows
are failed rather than made claimable with empty context.

Application queries always include the authenticated tenant identity. Existing
PostgreSQL row-level-security settings and policies are preserved by migration
004; policy design is deployment-specific. Use a server-only database role
whose ownership, grants, and RLS policies permit the required scoped queries.
Never expose that role, its connection string, or direct database access to API
clients.

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

CI runs those checks on Python 3.12 and applies migrations 001, 002, and 004 to
a PostgreSQL 16 service with pgvector before running the full suite. The
integration connection owns the tables and therefore bypasses the policy-free
RLS state created by migration 001. This validates queue semantics, not a
production deployment's role grants or deployment-specific RLS policies; test
those separately in staging.

## Docker

The shared API/worker image runs as an unprivileged user. It intentionally has
no image-level healthcheck because workers do not expose the API endpoints.

```bash
docker build -t rlm-api .
docker run --env-file .env -p 8080:8080 rlm-api
docker run --env-file .env rlm-api python worker.py
```

Run the API and worker as separate services against the same database. The
default command is the API. Configure HTTP `/livez` and `/healthz` probes on API
services. Configure process-level liveness and restart policy for worker
services.

## Deployment checklist

1. Select and verify current provider model IDs and credentials.
2. Ensure embedding output and database vector dimensions match.
3. Generate long random API keys and map each to exactly one tenant.
4. Keep `DATABASE_URL` and provider/API keys server-only.
5. Apply migrations 001 through 004 and schedule migration 003 cleanup.
6. Deploy at least one API and one worker (or explicitly enable embedded mode).
7. Configure termination grace longer than the maximum run timeout.
8. Set `WORKER_MAX_POLL_FAILURES` for the orchestrator restart/alert policy.
   Tune `WORKER_RECOVERY_BATCH_SIZE` (default 100) to bound stale-run locking.
9. Probe API `/livez` and `/healthz`; monitor worker processes separately.
10. Alert on exhausted runs, stale recovery, pending/aged summary outbox rows,
    and retry/failure logs.
11. Back up PostgreSQL according to the chosen provider's recovery policy.

Basic latency, attempt, and active root/sub-model metadata are stored in
`rlm_runs.usage`; state changes and retry metadata are stored in `rlm_events`.
Monitor `rlm_summary_outbox` rows with `completed_at IS NULL`, especially high
attempt counts or old `available_at`/`claimed_at` timestamps.
Run and summary embedding model/dimension fields are matched to each worker so
rolling deployments never mix vector spaces; mismatched summaries remain in
the outbox for a compatible worker.
