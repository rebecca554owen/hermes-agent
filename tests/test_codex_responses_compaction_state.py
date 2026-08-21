"""Native Responses compaction: durable table columns + CAS accessors.

Behavior contracts for the two new columns (``sessions.codex_responses_compaction_state``
and ``messages.codex_output_items``) and the revisioned compare-and-set accessor —
how the ledger must relate to the transcript, not snapshots of any one value.
"""

import json
import sqlite3

import pytest

from agent.responses_compaction import (
    NativeCompactionLedger,
    compaction_checkpoint_digest,
)
from hermes_state import (
    CodexResponsesCompactionStateConflictError,
    SessionDB,
)
from hermes_state_common import SCHEMA_SQL

# A canonical, credential-free route identity (must match
# normalize_compaction_endpoint's stable form).
ROUTE = {
    "issuer_kind": "codex_backend",
    "endpoint": "https://api.openai.com/v1",
    "model": "gpt-5.2",
}


def _sidecar(encrypted_content: str = "enc-blob-1") -> list:
    """A valid ordered compaction output sidecar, stamped for ROUTE."""
    return [
        {
            "type": "compaction",
            "encrypted_content": encrypted_content,
            "_issuer_kind": ROUTE["issuer_kind"],
            "_compaction_route": dict(ROUTE),
        }
    ]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return SessionDB(db_path=tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_sql_declares_compaction_columns():
    assert "    codex_responses_compaction_state TEXT," in SCHEMA_SQL
    assert "    codex_output_items TEXT," in SCHEMA_SQL


def test_fresh_db_has_compaction_columns(db, tmp_path):
    with sqlite3.connect(tmp_path / "state.db") as conn:
        session_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)")
        }
        message_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(messages)")
        }
    assert "codex_responses_compaction_state" in session_cols
    assert "codex_output_items" in message_cols


def test_existing_db_auto_migrates_compaction_columns(tmp_path, monkeypatch):
    """A pre-existing database without the columns is upgraded in place by
    the declarative column reconcile, preserving existing rows."""
    db_path = tmp_path / "state.db"
    old_schema = (
        SCHEMA_SQL.replace("    codex_responses_compaction_state TEXT,\n", "")
        .replace("    codex_output_items TEXT,\n", "")
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(old_schema)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('legacy-session', 'test', 0)"
        )
        conn.commit()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        session_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)")
        }
        message_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(messages)")
        }
    assert "codex_responses_compaction_state" in session_cols
    assert "codex_output_items" in message_cols
    assert db.get_session("legacy-session") is not None


# ---------------------------------------------------------------------------
# CAS accessors
# ---------------------------------------------------------------------------


def test_get_state_returns_empty_ledger_for_missing_session(db):
    state = db.get_codex_responses_compaction_state("no-such-session")
    assert state == NativeCompactionLedger.empty().to_dict()


def test_cas_advances_revision(db):
    key = db.create_session("cas-advance", "test")
    assert db.compare_and_set_codex_responses_compaction_state(
        key, expected_revision=0, state=NativeCompactionLedger.empty().to_dict()
    ) is True
    state = db.get_codex_responses_compaction_state(key)
    assert state["revision"] == 1
    assert state["routes"] == {}


def test_cas_stale_expected_revision_returns_false(db):
    key = db.create_session("cas-stale", "test")
    empty = NativeCompactionLedger.empty().to_dict()
    assert db.compare_and_set_codex_responses_compaction_state(
        key, expected_revision=0, state=empty
    ) is True
    # A concurrent writer already advanced the ledger: the stale CAS loses.
    assert db.compare_and_set_codex_responses_compaction_state(
        key, expected_revision=0, state=empty
    ) is False
    assert db.get_codex_responses_compaction_state(key)["revision"] == 1


def test_cas_rejects_invalid_expected_revision(db):
    key = db.create_session("cas-bad-rev", "test")
    empty = NativeCompactionLedger.empty().to_dict()
    for bad in (-1, True):
        with pytest.raises(ValueError):
            db.compare_and_set_codex_responses_compaction_state(
                key, expected_revision=bad, state=empty
            )


def test_cas_rejects_candidate_revision_mismatch(db):
    key = db.create_session("cas-bad-state", "test")
    with pytest.raises(ValueError):
        db.compare_and_set_codex_responses_compaction_state(
            key, expected_revision=0, state={"version": 3, "revision": 5, "routes": {}}
        )


# ---------------------------------------------------------------------------
# Message write path
# ---------------------------------------------------------------------------


def test_append_message_persists_codex_output_items(db):
    key = db.create_session("sidecar-write", "test")
    sidecar = _sidecar()
    db.append_message(
        key,
        "assistant",
        content="compacted prefix",
        codex_output_items=sidecar,
    )
    rows = db.get_messages_as_conversation(key)
    assistant = next(r for r in rows if r["role"] == "assistant")
    assert assistant["codex_output_items"] == sidecar


def test_append_message_checkpoint_commits_ledger(db):
    key = db.create_session("checkpoint-commit", "test")
    sidecar = _sidecar()
    digest = compaction_checkpoint_digest(sidecar)
    policy = {
        "route": dict(ROUTE),
        "capability": "item_observed",
        "revision": 0,
        "compaction_count": 1,
        "fallback_count": 0,
        "last_compaction_digest": digest,
        "last_error": None,
    }
    db.append_message(
        key,
        "assistant",
        content="compacted prefix",
        codex_output_items=sidecar,
        codex_responses_compaction_policy=policy,
        expected_codex_responses_compaction_revision=0,
    )
    state = db.get_codex_responses_compaction_state(key)
    assert state["revision"] == 1
    entry = next(iter(state["routes"].values()))
    assert entry["capability"] == "item_observed"
    assert entry["compaction_count"] == 1
    assert entry["last_compaction_digest"] == digest


def test_append_message_checkpoint_stale_revision_raises(db):
    key = db.create_session("checkpoint-stale", "test")
    sidecar = _sidecar()
    digest = compaction_checkpoint_digest(sidecar)
    policy = {
        "route": dict(ROUTE),
        "capability": "item_observed",
        "revision": 0,
        "compaction_count": 1,
        "fallback_count": 0,
        "last_compaction_digest": digest,
        "last_error": None,
    }
    db.append_message(
        key,
        "assistant",
        content="compacted prefix",
        codex_output_items=sidecar,
        codex_responses_compaction_policy=policy,
        expected_codex_responses_compaction_revision=0,
    )
    # Ledger is now at revision 1 — a stale checkpoint must fail closed.
    with pytest.raises(CodexResponsesCompactionStateConflictError):
        db.append_message(
            key,
            "assistant",
            content="another checkpoint",
            codex_output_items=_sidecar("enc-blob-2"),
            codex_responses_compaction_policy={
                **policy,
                "revision": 0,
                "last_compaction_digest": compaction_checkpoint_digest(
                    _sidecar("enc-blob-2")
                ),
            },
            expected_codex_responses_compaction_revision=0,
        )


def test_append_message_rejects_mismatched_checkpoint_sidecar(db):
    key = db.create_session("checkpoint-mismatch", "test")
    policy = {
        "route": dict(ROUTE),
        "capability": "item_observed",
        "revision": 0,
        "compaction_count": 1,
        "fallback_count": 0,
        "last_compaction_digest": compaction_checkpoint_digest(_sidecar("other")),
        "last_error": None,
    }
    with pytest.raises(ValueError):
        db.append_message(
            key,
            "assistant",
            content="compacted prefix",
            codex_output_items=_sidecar("enc-blob-1"),
            codex_responses_compaction_policy=policy,
            expected_codex_responses_compaction_revision=0,
        )


def test_append_message_requires_checkpoint_args_together(db):
    key = db.create_session("checkpoint-args", "test")
    with pytest.raises(ValueError):
        db.append_message(
            key,
            "assistant",
            content="x",
            codex_responses_compaction_policy={},
        )
