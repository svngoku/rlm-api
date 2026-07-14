-- Durable queue fields for API/worker separation. Safe to run repeatedly.
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

-- Existing pre-004 rows may only have corpus_ref. New runs always store context inline.
UPDATE rlm_runs
SET context = COALESCE(context, corpus_ref->>'context', '')
WHERE context IS NULL;

ALTER TABLE rlm_runs
  ALTER COLUMN context SET NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'rlm_runs_status_check'
  ) THEN
    ALTER TABLE rlm_runs
      ADD CONSTRAINT rlm_runs_status_check
      CHECK (status IN ('queued', 'running', 'succeeded', 'failed'));
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'rlm_runs_attempts_check'
  ) THEN
    ALTER TABLE rlm_runs
      ADD CONSTRAINT rlm_runs_attempts_check
      CHECK (attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts);
  END IF;
END
$$;

CREATE INDEX IF NOT EXISTS rlm_runs_queue_idx
  ON rlm_runs (available_at, created_at)
  WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS rlm_runs_stale_worker_idx
  ON rlm_runs (worker_heartbeat_at)
  WHERE status = 'running';

CREATE INDEX IF NOT EXISTS rlm_runs_tenant_lookup_idx
  ON rlm_runs (tenant_id, id);

CREATE INDEX IF NOT EXISTS rlm_events_run_created_idx
  ON rlm_events (run_id, created_at);

-- The service uses one server-only database role and enforces tenant scope in every
-- application query. 001 enabled RLS without policies, which is misleading and can
-- lock out a non-owner service role. Direct database/client access is unsupported.
ALTER TABLE memory_items DISABLE ROW LEVEL SECURITY;
ALTER TABLE rlm_runs DISABLE ROW LEVEL SECURITY;
ALTER TABLE rlm_events DISABLE ROW LEVEL SECURITY;
