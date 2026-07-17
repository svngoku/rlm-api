-- Durable queue fields for API/worker separation. Safe to run repeatedly.
-- This migration is psql-only. Run with autocommit and do not wrap it in an
-- explicit transaction because indexes are built and repaired CONCURRENTLY.
ALTER TABLE rlm_runs
  ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default',
  ADD COLUMN IF NOT EXISTS context TEXT,
  ADD COLUMN IF NOT EXISTS include_trajectory BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS trajectory JSONB,
  ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 3,
  ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS worker_id TEXT,
  ADD COLUMN IF NOT EXISTS worker_heartbeat_at TIMESTAMPTZ;

ALTER TABLE memory_items
  ADD COLUMN IF NOT EXISTS source_run_id UUID;

-- Recover only genuine inline legacy context.
UPDATE rlm_runs
SET context = corpus_ref->>'context',
    updated_at = now()
WHERE (context IS NULL OR btrim(context) = '')
  AND NULLIF(btrim(corpus_ref->>'context'), '') IS NOT NULL;

-- Never make an unrecoverable nonterminal run claimable with an empty corpus.
UPDATE rlm_runs
SET status = 'failed',
    error = 'migration_missing_context',
    completed_at = COALESCE(completed_at, now()),
    updated_at = now(),
    worker_id = NULL,
    worker_heartbeat_at = NULL
WHERE status IN ('queued', 'running')
  AND (context IS NULL OR btrim(context) = '');

-- Empty context is retained only as historical data on terminal legacy rows.
UPDATE rlm_runs
SET context = ''
WHERE status IN ('succeeded', 'failed')
  AND context IS NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'rlm_runs_context_not_null'
      AND conrelid = 'rlm_runs'::regclass
  ) THEN
    ALTER TABLE rlm_runs
      ADD CONSTRAINT rlm_runs_context_not_null
      CHECK (context IS NOT NULL) NOT VALID;
  END IF;
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'rlm_runs_status_check'
      AND conrelid = 'rlm_runs'::regclass
  ) THEN
    ALTER TABLE rlm_runs
      ADD CONSTRAINT rlm_runs_status_check
      CHECK (status IN ('queued', 'running', 'succeeded', 'failed'));
  END IF;
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'rlm_runs_attempts_check'
      AND conrelid = 'rlm_runs'::regclass
  ) THEN
    ALTER TABLE rlm_runs
      ADD CONSTRAINT rlm_runs_attempts_check
      CHECK (attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts);
  END IF;
END
$$;

ALTER TABLE rlm_runs
  VALIDATE CONSTRAINT rlm_runs_context_not_null;

ALTER TABLE rlm_runs
  ALTER COLUMN context SET NOT NULL;

-- A failed concurrent build leaves an invalid catalog entry that IF NOT EXISTS
-- would otherwise preserve. Generate schema-qualified concurrent drops for
-- invalid remnants owned by the tables and index names managed below.
SELECT format(
         'DROP INDEX CONCURRENTLY IF EXISTS %s',
         format('%I.%I', index_namespace.nspname, index_class.relname)
       )
FROM pg_index AS index_state
JOIN pg_class AS index_class
  ON index_class.oid = index_state.indexrelid
JOIN pg_namespace AS index_namespace
  ON index_namespace.oid = index_class.relnamespace
WHERE NOT index_state.indisvalid
  AND (
    (
      index_state.indrelid = to_regclass('rlm_runs')
      AND index_class.relname IN (
        'rlm_runs_queue_idx',
        'rlm_runs_stale_worker_idx',
        'rlm_runs_tenant_lookup_idx'
      )
    )
    OR (
      index_state.indrelid = to_regclass('rlm_events')
      AND index_class.relname = 'rlm_events_run_created_idx'
    )
    OR (
      index_state.indrelid = to_regclass('rlm_summary_outbox')
      AND index_class.relname = 'rlm_summary_outbox_pending_idx'
    )
    OR (
      index_state.indrelid = to_regclass('memory_items')
      AND index_class.relname = 'memory_run_summary_source_idx'
    )
  )
ORDER BY index_namespace.nspname, index_class.relname
\gexec

CREATE INDEX CONCURRENTLY IF NOT EXISTS rlm_runs_queue_idx
  ON rlm_runs (available_at, created_at)
  WHERE status = 'queued';

CREATE INDEX CONCURRENTLY IF NOT EXISTS rlm_runs_stale_worker_idx
  ON rlm_runs (worker_heartbeat_at)
  WHERE status = 'running';

CREATE INDEX CONCURRENTLY IF NOT EXISTS rlm_runs_tenant_lookup_idx
  ON rlm_runs (tenant_id, id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS rlm_events_run_created_idx
  ON rlm_events (run_id, created_at);

CREATE TABLE IF NOT EXISTS rlm_summary_outbox (
  run_id       UUID PRIMARY KEY REFERENCES rlm_runs(id) ON DELETE CASCADE,
  tenant_id    TEXT NOT NULL,
  subject_id   TEXT NOT NULL,
  namespace    TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
  content      TEXT NOT NULL,
  metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
  attempts     INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  claimed_by   TEXT,
  claimed_at   TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  last_error   TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE rlm_summary_outbox ENABLE ROW LEVEL SECURITY;

ALTER TABLE rlm_summary_outbox
  ADD COLUMN IF NOT EXISTS embedding_model TEXT,
  ADD COLUMN IF NOT EXISTS embedding_dim INTEGER;

UPDATE rlm_summary_outbox AS outbox
SET embedding_model = NULLIF(btrim(run.model_config->>'embedding_model'), ''),
    embedding_dim = CASE
      WHEN run.model_config->>'embedding_dim' ~ '^[1-9][0-9]{0,8}$'
      THEN (run.model_config->>'embedding_dim')::INTEGER
      ELSE NULL
    END,
    updated_at = now()
FROM rlm_runs AS run
WHERE run.id = outbox.run_id
  AND (outbox.embedding_model IS NULL OR outbox.embedding_dim IS NULL);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'rlm_summary_outbox_embedding_not_null'
      AND conrelid = 'rlm_summary_outbox'::regclass
  ) THEN
    ALTER TABLE rlm_summary_outbox
      ADD CONSTRAINT rlm_summary_outbox_embedding_not_null
      CHECK (
        embedding_model IS NOT NULL
        AND embedding_dim IS NOT NULL
        AND embedding_dim > 0
      ) NOT VALID;
  END IF;
END
$$;

ALTER TABLE rlm_summary_outbox
  VALIDATE CONSTRAINT rlm_summary_outbox_embedding_not_null;

ALTER TABLE rlm_summary_outbox
  ALTER COLUMN embedding_model SET NOT NULL,
  ALTER COLUMN embedding_dim SET NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS rlm_summary_outbox_pending_idx
  ON rlm_summary_outbox (available_at, created_at)
  WHERE completed_at IS NULL;

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS memory_run_summary_source_idx
  ON memory_items (tenant_id, subject_id, namespace, source_run_id)
  WHERE source_run_id IS NOT NULL;
