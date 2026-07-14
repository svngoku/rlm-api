-- Run this periodically (pg_cron, Neon scheduled query, or external cron)
-- Removes expired memory items to keep the table lean.
DELETE FROM memory_items
WHERE expires_at IS NOT NULL
  AND expires_at < now();
