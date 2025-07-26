-- Initialise database schema for rag-industrial-eval-guardrails
-- Creates the necessary tables and indexes for document and chunk storage

-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for potential trigram indexing (optional)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Table storing high level documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id TEXT UNIQUE NOT NULL,
    doc_name TEXT NOT NULL,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table storing chunks of documents with their embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL,
    content TEXT,
    embedding vector(1536),
    tsv tsvector,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (document_id, chunk_id)
);

-- Indexes for vector similarity search and full text search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);