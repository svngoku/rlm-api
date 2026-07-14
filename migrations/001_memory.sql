-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Memory kind enum
CREATE TYPE memory_kind AS ENUM (
  'fact',
  'preference',
  'decision',
  'episode',
  'run_summary',
  'feedback'
);

-- Core memory table
CREATE TABLE memory_items (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id        TEXT NOT NULL,
  subject_id       TEXT NOT NULL,
  namespace        TEXT NOT NULL DEFAULT 'default',
  kind             memory_kind NOT NULL,
  content          TEXT NOT NULL,
  metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding        vector(1536),
  importance       REAL NOT NULL DEFAULT 0.5
                   CHECK (importance >= 0 AND importance <= 1),
  access_count     INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TIMESTAMPTZ,
  expires_at       TIMESTAMPTZ,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  content_tsv      TSVECTOR GENERATED ALWAYS AS (
                     to_tsvector('english', content)
                   ) STORED
);

CREATE INDEX memory_scope_idx
  ON memory_items (tenant_id, subject_id, namespace, created_at DESC);

CREATE INDEX memory_fts_idx
  ON memory_items USING GIN (content_tsv);

CREATE INDEX memory_embedding_hnsw_idx
  ON memory_items USING hnsw (embedding vector_cosine_ops)
  WHERE embedding IS NOT NULL;

CREATE INDEX memory_expiry_idx
  ON memory_items (expires_at)
  WHERE expires_at IS NOT NULL;

-- RLM runs table
CREATE TABLE rlm_runs (
  id              UUID PRIMARY KEY,
  tenant_id       TEXT NOT NULL,
  subject_id      TEXT NOT NULL,
  status          TEXT NOT NULL,
  task            TEXT NOT NULL,
  corpus_ref      JSONB,
  memory_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb,
  model_config    JSONB NOT NULL,
  limits          JSONB NOT NULL,
  result          JSONB,
  usage           JSONB NOT NULL DEFAULT '{}'::jsonb,
  error           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at      TIMESTAMPTZ,
  completed_at    TIMESTAMPTZ
);

CREATE INDEX rlm_runs_scope_idx
  ON rlm_runs (tenant_id, subject_id, created_at DESC);

-- RLM trajectory events
CREATE TABLE rlm_events (
  id         BIGSERIAL PRIMARY KEY,
  run_id     UUID NOT NULL REFERENCES rlm_runs(id) ON DELETE CASCADE,
  sequence   INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  payload    JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, sequence)
);

-- Row Level Security (activate when using Neon Data API or direct client access)
ALTER TABLE memory_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE rlm_runs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE rlm_events   ENABLE ROW LEVEL SECURITY;
