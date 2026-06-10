-- Runs once when the Postgres volume is first created.
-- Creates the langfuse database (the langfuse container expects it to exist)
-- and enables pgvector on the main DB.

CREATE DATABASE langfuse;

\connect claims_copilot
CREATE EXTENSION IF NOT EXISTS vector;

\connect langfuse
CREATE EXTENSION IF NOT EXISTS pgcrypto;
