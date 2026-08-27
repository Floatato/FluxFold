PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS embedding_model_signatures (
    model_signature_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    dtype TEXT NOT NULL CHECK (dtype = 'float32'),
    normalization TEXT NOT NULL,
    query_mode TEXT NOT NULL,
    document_mode TEXT NOT NULL,
    signature_hash TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS memory_spaces (
    memory_space_id TEXT PRIMARY KEY,
    space_key TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS memory_space_model_signatures (
    memory_space_id TEXT NOT NULL REFERENCES memory_spaces(memory_space_id) ON DELETE RESTRICT,
    purpose TEXT NOT NULL CHECK (purpose IN ('retrieval', 'boundary')),
    model_signature_id TEXT NOT NULL REFERENCES embedding_model_signatures(model_signature_id) ON DELETE RESTRICT,
    activated_at INTEGER NOT NULL,
    PRIMARY KEY (memory_space_id, purpose)
) STRICT;

CREATE TABLE IF NOT EXISTS situational_episodes (
    episode_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL REFERENCES memory_spaces(memory_space_id) ON DELETE RESTRICT,
    source_type TEXT NOT NULL,
    source_key TEXT NOT NULL,
    source_sequence INTEGER NOT NULL CHECK (source_sequence >= 0),
    payload_version TEXT NOT NULL,
    source_started_at INTEGER,
    source_ended_at INTEGER,
    source_timezone TEXT,
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
    source_chars INTEGER NOT NULL CHECK (source_chars >= 0),
    created_at INTEGER NOT NULL,
    UNIQUE (memory_space_id, source_type, source_key),
    UNIQUE (memory_space_id, source_sequence)
) STRICT;

CREATE TABLE IF NOT EXISTS episode_blocks (
    block_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES situational_episodes(episode_id) ON DELETE RESTRICT,
    sequence_no INTEGER NOT NULL CHECK (sequence_no >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    message_phase TEXT CHECK (message_phase IS NULL OR message_phase = 'final'),
    speaker_id TEXT,
    speaker_name TEXT,
    content TEXT NOT NULL,
    observed_at INTEGER,
    metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
    preprocessor_version TEXT,
    is_truncated INTEGER NOT NULL DEFAULT 0 CHECK (is_truncated IN (0, 1)),
    truncation_json TEXT CHECK (truncation_json IS NULL OR json_valid(truncation_json)),
    UNIQUE (episode_id, sequence_no)
) STRICT;

CREATE TABLE IF NOT EXISTS domain_operations (
    operation_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL REFERENCES memory_spaces(memory_space_id) ON DELETE RESTRICT,
    operation_type TEXT NOT NULL CHECK (operation_type IN ('add_episode_memories', 'review_subject', 'split_subject', 'refresh_subject_summary', 'retire_memory')),
    actor TEXT NOT NULL,
    config_signature TEXT NOT NULL,
    reason TEXT NOT NULL,
    committed_at INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS memory_units (
    memory_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL REFERENCES memory_spaces(memory_space_id) ON DELETE RESTRICT,
    lifecycle_status TEXT NOT NULL CHECK (lifecycle_status IN ('active', 'retired')),
    latest_source_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    retired_at INTEGER,
    retired_by_operation_id TEXT REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    CHECK ((lifecycle_status = 'active' AND retired_at IS NULL AND retired_by_operation_id IS NULL)
        OR (lifecycle_status = 'retired' AND retired_at IS NOT NULL AND retired_by_operation_id IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS memory_versions (
    memory_version_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_units(memory_id) ON DELETE RESTRICT,
    version_no INTEGER NOT NULL CHECK (version_no >= 1),
    is_latest INTEGER NOT NULL CHECK (is_latest IN (0, 1)),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
    char_count INTEGER NOT NULL CHECK (char_count >= 0),
    created_by_operation_id TEXT NOT NULL REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    created_at INTEGER NOT NULL,
    UNIQUE (memory_id, version_no)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_latest_version
ON memory_versions(memory_id) WHERE is_latest = 1;

CREATE TABLE IF NOT EXISTS memory_version_provenance (
    memory_version_id TEXT NOT NULL REFERENCES memory_versions(memory_version_id) ON DELETE RESTRICT,
    episode_id TEXT NOT NULL REFERENCES situational_episodes(episode_id) ON DELETE RESTRICT,
    PRIMARY KEY (memory_version_id, episode_id)
) STRICT;

CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL REFERENCES memory_spaces(memory_space_id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    summary TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL CHECK (lifecycle_status IN ('active', 'retired')),
    new_memory_count INTEGER NOT NULL CHECK (new_memory_count >= 0),
    summary_revision INTEGER NOT NULL CHECK (summary_revision >= 0),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    retired_at INTEGER,
    retired_by_operation_id TEXT REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    CHECK ((lifecycle_status = 'active' AND retired_at IS NULL AND retired_by_operation_id IS NULL)
        OR (lifecycle_status = 'retired' AND retired_at IS NOT NULL AND retired_by_operation_id IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS subject_memory_links (
    link_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE RESTRICT,
    memory_id TEXT NOT NULL REFERENCES memory_units(memory_id) ON DELETE RESTRICT,
    link_basis TEXT NOT NULL CHECK (link_basis IN ('direct', 'contextual')),
    linked_at INTEGER NOT NULL,
    unlinked_at INTEGER,
    opened_by_operation_id TEXT NOT NULL REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    closed_by_operation_id TEXT REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    CHECK ((unlinked_at IS NULL AND closed_by_operation_id IS NULL)
        OR (unlinked_at IS NOT NULL AND closed_by_operation_id IS NOT NULL))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_subject_memory_active_link
ON subject_memory_links(subject_id, memory_id) WHERE unlinked_at IS NULL;

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_version_id TEXT NOT NULL REFERENCES memory_versions(memory_version_id) ON DELETE RESTRICT,
    model_signature_id TEXT NOT NULL REFERENCES embedding_model_signatures(model_signature_id) ON DELETE RESTRICT,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    source_hash TEXT NOT NULL CHECK (length(source_hash) = 64),
    vector BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (memory_version_id, model_signature_id),
    CHECK (length(vector) = dimension * 4)
) STRICT;

CREATE TABLE IF NOT EXISTS subject_embeddings (
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE RESTRICT,
    embedding_kind TEXT NOT NULL CHECK (embedding_kind IN ('name', 'name_summary')),
    model_signature_id TEXT NOT NULL REFERENCES embedding_model_signatures(model_signature_id) ON DELETE RESTRICT,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    source_hash TEXT NOT NULL CHECK (length(source_hash) = 64),
    vector BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (subject_id, embedding_kind, model_signature_id),
    CHECK (length(vector) = dimension * 4)
) STRICT;

CREATE TABLE IF NOT EXISTS episode_extractions (
    episode_id TEXT PRIMARY KEY REFERENCES situational_episodes(episode_id) ON DELETE RESTRICT,
    input_hash TEXT NOT NULL CHECK (length(input_hash) = 64),
    extractor_config_signature TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'terminal_failure')),
    completed_operation_id TEXT REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    completed_at INTEGER NOT NULL,
    error_class TEXT,
    error_message TEXT,
    CHECK ((status = 'completed' AND error_class IS NULL AND error_message IS NULL)
        OR (status = 'terminal_failure' AND error_class IS NOT NULL AND error_message IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS domain_operation_effects (
    effect_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL REFERENCES domain_operations(operation_id) ON DELETE RESTRICT,
    object_type TEXT NOT NULL CHECK (object_type IN ('episode', 'memory', 'memory_version', 'subject', 'link')),
    object_id TEXT NOT NULL,
    effect_type TEXT NOT NULL,
    metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json))
) STRICT;

CREATE INDEX IF NOT EXISTS ix_episode_space_sequence ON situational_episodes(memory_space_id, source_sequence);
CREATE INDEX IF NOT EXISTS ix_memory_space_status ON memory_units(memory_space_id, lifecycle_status);
CREATE INDEX IF NOT EXISTS ix_memory_latest_source ON memory_units(memory_space_id, latest_source_at);
CREATE INDEX IF NOT EXISTS ix_memory_version_history ON memory_versions(memory_id, version_no);
CREATE INDEX IF NOT EXISTS ix_provenance_episode ON memory_version_provenance(episode_id, memory_version_id);
CREATE INDEX IF NOT EXISTS ix_subject_space_status ON subjects(memory_space_id, lifecycle_status);
CREATE INDEX IF NOT EXISTS ix_link_subject_active ON subject_memory_links(subject_id, unlinked_at, memory_id);
CREATE INDEX IF NOT EXISTS ix_link_memory_active ON subject_memory_links(memory_id, unlinked_at, subject_id);
CREATE INDEX IF NOT EXISTS ix_operation_space_time ON domain_operations(memory_space_id, committed_at, operation_id);

CREATE TRIGGER IF NOT EXISTS immutable_episode_update
BEFORE UPDATE ON situational_episodes
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'situational episodes are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_episode_delete
BEFORE DELETE ON situational_episodes
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'situational episodes are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_block_update
BEFORE UPDATE ON episode_blocks
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'episode blocks are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_block_delete
BEFORE DELETE ON episode_blocks
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'episode blocks are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_historical_version_update
BEFORE UPDATE ON memory_versions
WHEN fluxfold_management_mode() = 0
 AND NOT (OLD.is_latest = 1 AND NEW.is_latest = 0
          AND OLD.memory_version_id = NEW.memory_version_id
          AND OLD.memory_id = NEW.memory_id
          AND OLD.version_no = NEW.version_no
          AND OLD.content = NEW.content
          AND OLD.content_hash = NEW.content_hash
          AND OLD.char_count = NEW.char_count
          AND OLD.created_by_operation_id = NEW.created_by_operation_id
          AND OLD.created_at = NEW.created_at)
BEGIN SELECT RAISE(ABORT, 'memory versions are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_version_delete
BEFORE DELETE ON memory_versions
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'memory versions are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_provenance_update
BEFORE UPDATE ON memory_version_provenance
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'provenance rows are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_provenance_delete
BEFORE DELETE ON memory_version_provenance
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'provenance rows are immutable'); END;

CREATE TRIGGER IF NOT EXISTS immutable_operation_update
BEFORE UPDATE ON domain_operations
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'domain operations are append-only'); END;

CREATE TRIGGER IF NOT EXISTS immutable_operation_delete
BEFORE DELETE ON domain_operations
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'domain operations are append-only'); END;

CREATE TRIGGER IF NOT EXISTS immutable_effect_update
BEFORE UPDATE ON domain_operation_effects
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'domain operation effects are append-only'); END;

CREATE TRIGGER IF NOT EXISTS immutable_effect_delete
BEFORE DELETE ON domain_operation_effects
WHEN fluxfold_management_mode() = 0
BEGIN SELECT RAISE(ABORT, 'domain operation effects are append-only'); END;
