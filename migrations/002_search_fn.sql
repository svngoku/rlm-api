-- Hybrid RRF search: semantic (pgvector) + keyword (FTS)
CREATE OR REPLACE FUNCTION search_memories(
  p_tenant_id       TEXT,
  p_subject_id      TEXT,
  p_namespace       TEXT,
  p_query           TEXT,
  p_embedding       vector(1536),
  p_limit           INTEGER DEFAULT 8,
  p_semantic_weight REAL DEFAULT 1.0,
  p_keyword_weight  REAL DEFAULT 0.7
)
RETURNS TABLE (
  id       UUID,
  kind     memory_kind,
  content  TEXT,
  metadata JSONB,
  score    REAL
)
LANGUAGE sql
STABLE
AS $$
WITH scoped AS (
  SELECT *
  FROM memory_items
  WHERE tenant_id = p_tenant_id
    AND subject_id = p_subject_id
    AND namespace  = p_namespace
    AND (expires_at IS NULL OR expires_at > now())
),
semantic AS (
  SELECT id,
         row_number() OVER (ORDER BY embedding <=> p_embedding) AS rank
  FROM scoped
  WHERE embedding IS NOT NULL
  ORDER BY embedding <=> p_embedding
  LIMIT 50
),
keyword AS (
  SELECT id,
         row_number() OVER (
           ORDER BY ts_rank_cd(
             content_tsv,
             websearch_to_tsquery('english', p_query)
           ) DESC
         ) AS rank
  FROM scoped
  WHERE content_tsv @@ websearch_to_tsquery('english', p_query)
  LIMIT 50
),
fused AS (
  SELECT
    COALESCE(s.id, k.id) AS id,
    COALESCE(p_semantic_weight / (60.0 + s.rank), 0) +
    COALESCE(p_keyword_weight  / (60.0 + k.rank), 0) AS rrf_score
  FROM semantic s
  FULL OUTER JOIN keyword k USING (id)
)
SELECT
  m.id,
  m.kind,
  m.content,
  m.metadata,
  (f.rrf_score * (0.8 + 0.2 * m.importance))::REAL AS score
FROM fused f
JOIN memory_items m ON m.id = f.id
ORDER BY score DESC
LIMIT p_limit;
$$;
