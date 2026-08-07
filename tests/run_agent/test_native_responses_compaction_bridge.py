"""Unit tests for the ported native Responses compaction policy primitives
and the local-summary bridge.

Covers the dependency-free policy/ledger layer in ``agent/responses_compaction.py``:
the capability state machine, route normalization, checkpoint digests, request
override validation, and the ``session_db=None`` graceful paths. The durable
``session_db``-backed persistence paths (``get_codex_responses_compaction_state`` /
``compare_and_set_codex_responses_compaction_state``) belong to a later task and
are not exercised here.

The second half covers the local-summary bridge: wrapping a local handoff
summary into a Responses compaction input item (``wrap_summary_as_compaction_item``
/ ``unwrap_compaction_item``), stamping it for sidecar custody
(``compaction_item_to_sidecar``), and the ``ContextCompressor.compress()`` mount
gate driven by ``compression.remote`` (auto/on/off).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_input_items,
)
from agent.context_compressor import ContextCompressor
from agent.responses_compaction import (
    ALL_CAPABILITY_STATES,
    BRIDGE_ENVELOPE_PREFIX,
    CAPABILITY_STATES,
    CHECKPOINT_REQUIRED_CAPABILITY_STATES,
    EVICTABLE_CAPABILITY_STATES,
    MAX_COMPACTION_LEDGER_ROUTES,
    NATIVE_RESPONSES_RESERVED_REQUEST_OVERRIDE_KEYS,
    OWNING_CAPABILITY_STATES,
    TERMINAL_STATES,
    NativeCompactionLedger,
    NativeCompactionPolicy,
    NativeCompactionRoute,
    NativeCompactionStateError,
    _BRIDGE_ENVELOPE_VERSION,
    build_native_request_overrides,
    compaction_checkpoint_digest,
    compaction_item_to_sidecar,
    compaction_route_key,
    load_compaction_ledger,
    load_policy_for_route,
    normalize_compaction_endpoint,
    persist_policy_compare_and_set,
    read_policy_for_route,
    route_for_request,
    should_defer_automatic_hermes_compaction,
    unwrap_compaction_item,
    validate_native_request_overrides,
    validate_responses_continuation_overrides,
    wrap_summary_as_compaction_item,
)


def _openai_route() -> NativeCompactionRoute:
    return NativeCompactionRoute(
        issuer_kind="openai",
        endpoint="https://api.openai.com/v1",
        model="gpt-5.1-codex",
    )


def _route(index: int) -> NativeCompactionRoute:
    """Distinct route for ledger capacity/eviction scenarios."""
    return NativeCompactionRoute(
        issuer_kind="openai",
        endpoint="https://api.openai.com/v1",
        model=f"gpt-5.1-codex-{index:02d}",
    )


def test_ledger_state_transition():
    ledger = NativeCompactionLedger.empty()
    assert ledger.revision == 0
    assert ledger.routes == {}

    route = _openai_route()
    initial = ledger.policy_for(route)
    # Unobserved routes start at the unknown capability.
    assert initial.capability == "unknown"
    assert initial.route == route
    assert initial.revision == 0

    accepted = initial.transition("shape_accepted")
    assert accepted.capability == "shape_accepted"
    assert accepted is not initial
    assert "shape_accepted" in CAPABILITY_STATES

    # with_policy records the route; policy_for re-reads the recorded state.
    next_ledger = ledger.with_policy(accepted)
    assert next_ledger is not ledger
    assert set(next_ledger.routes) == {compaction_route_key(route)}
    assert next_ledger.policy_for(route).capability == "shape_accepted"
    # The original ledger is immutable: recording mutated nothing in it.
    assert ledger.routes == {}
    # The capability state machine never regresses.
    with pytest.raises(ValueError):
        next_ledger.policy_for(route).transition("unknown")

    # Ledger serialization round-trips the capability state.
    restored = NativeCompactionLedger.from_dict(next_ledger.to_dict())
    assert restored.policy_for(route).capability == "shape_accepted"


def test_ledger_evicts_evictable_route_at_capacity():
    # A full 32-route ledger holds one evictable discovery hint, one terminal,
    # and the rest ownership-bearing custody.
    evictable = _route(0)
    terminal = _route(1)
    owning = [_route(i) for i in range(2, MAX_COMPACTION_LEDGER_ROUTES)]
    routes = {
        compaction_route_key(evictable): NativeCompactionPolicy(
            route=evictable, capability="shape_accepted"
        ),
        compaction_route_key(terminal): NativeCompactionPolicy(
            route=terminal, capability="unsupported"
        ),
    }
    for item in owning:
        routes[compaction_route_key(item)] = NativeCompactionPolicy(
            route=item, capability="replay_verified"
        )
    ledger = NativeCompactionLedger(revision=0, routes=routes)
    assert len(ledger.routes) == MAX_COMPACTION_LEDGER_ROUTES

    # A 33rd route overflows: the only evictable hint is dropped while every
    # ownership-bearing route and terminal history survives. The freshly added
    # route is never evicted even though its own state is evictable.
    newcomer = _route(MAX_COMPACTION_LEDGER_ROUTES)
    next_ledger = ledger.with_policy(
        NativeCompactionPolicy(route=newcomer, capability="shape_accepted")
    )
    assert len(next_ledger.routes) == MAX_COMPACTION_LEDGER_ROUTES
    assert compaction_route_key(evictable) not in next_ledger.routes
    assert compaction_route_key(terminal) in next_ledger.routes
    for item in owning:
        assert compaction_route_key(item) in next_ledger.routes
    assert (
        next_ledger.routes[compaction_route_key(newcomer)].capability
        == "shape_accepted"
    )


def test_ledger_capacity_overflow_raises():
    # A full ledger of durable custody or terminal history has nothing
    # evictable: recording one more route must fail closed rather than
    # silently drop ownership or terminal state.
    routes = {}
    for i in range(MAX_COMPACTION_LEDGER_ROUTES):
        if i % 5 == 0:
            capability = "unsupported"
        elif i % 3 == 0:
            capability = "quarantined"
        else:
            capability = "replay_verified"
        item = _route(i)
        routes[compaction_route_key(item)] = NativeCompactionPolicy(
            route=item, capability=capability
        )
    ledger = NativeCompactionLedger(revision=0, routes=routes)
    assert len(ledger.routes) == MAX_COMPACTION_LEDGER_ROUTES
    assert not EVICTABLE_CAPABILITY_STATES.intersection(
        policy.capability for policy in ledger.routes.values()
    )

    with pytest.raises(
        NativeCompactionStateError, match="full of durable custody"
    ):
        ledger.with_policy(
            NativeCompactionPolicy(
                route=_route(MAX_COMPACTION_LEDGER_ROUTES),
                capability="shape_accepted",
            )
        )


def test_normalize_compaction_endpoint():
    # Credentials, query strings, and fragments are stripped; path is kept.
    assert (
        normalize_compaction_endpoint(
            "https://user:secret@api.openai.com/v1?key=value#frag"
        )
        == "https://api.openai.com/v1"
    )
    assert (
        normalize_compaction_endpoint(
            "https://user:secret@example.com:8443/path?x=1#y"
        )
        == "https://example.com:8443/path"
    )
    # Scheme-less network-path values drop userinfo and query.
    assert (
        normalize_compaction_endpoint(
            "user:secret@chatgpt.com/backend-api/codex?x=1"
        )
        == "//chatgpt.com/backend-api/codex"
    )
    # Empty and blank inputs normalize to the empty string.
    assert normalize_compaction_endpoint("") == ""
    assert normalize_compaction_endpoint("   ") == ""
    # Whitespace and control characters fail closed to the sentinel, never raise.
    assert normalize_compaction_endpoint("not a url") == "invalid://redacted"
    assert normalize_compaction_endpoint("\x01bad") == "invalid://redacted"
    # Malformed values (out-of-range port, junk) never raise and never leak userinfo.
    assert normalize_compaction_endpoint("user@host:99999") == "host:99999"
    assert normalize_compaction_endpoint("%%%") == "%%%"
    assert normalize_compaction_endpoint("http://") != ""


def test_compaction_checkpoint_digest_detects_tamper():
    def sidecar(summary):
        return [
            {
                "type": "compaction",
                "encrypted_content": "encrypted-payload",
                "summary_text": summary,
            }
        ]

    digest = compaction_checkpoint_digest(sidecar("original summary"))
    assert isinstance(digest, str) and len(digest) == 64
    # Deterministic for identical ordered checkpoints.
    assert compaction_checkpoint_digest(sidecar("original summary")) == digest
    # Any payload change (e.g. summary_text) changes the digest.
    assert compaction_checkpoint_digest(sidecar("tampered summary")) != digest

    # Non-compaction or malformed item lists yield no digest, never raise.
    assert compaction_checkpoint_digest([]) is None
    assert compaction_checkpoint_digest(None) is None
    assert (
        compaction_checkpoint_digest(
            [{"type": "message", "role": "user", "content": "hi"}]
        )
        is None
    )
    assert (
        compaction_checkpoint_digest([{"type": "compaction", "encrypted_content": ""}])
        is None
    )


def test_capability_states():
    # Canonical state partitions per the module's invariants: terminal states
    # never overlap the live progression, and the full universe is exactly the
    # union of the two.
    assert TERMINAL_STATES.isdisjoint(CAPABILITY_STATES)
    assert ALL_CAPABILITY_STATES == set(CAPABILITY_STATES) | TERMINAL_STATES

    # Ownership, checkpoint, and eviction partitions per the source definitions:
    # evictable hints never overlap ownership or checkpoint custody, and every
    # checkpoint-required state is ownership-bearing.
    assert EVICTABLE_CAPABILITY_STATES.isdisjoint(OWNING_CAPABILITY_STATES)
    assert EVICTABLE_CAPABILITY_STATES.isdisjoint(CHECKPOINT_REQUIRED_CAPABILITY_STATES)
    assert CHECKPOINT_REQUIRED_CAPABILITY_STATES <= OWNING_CAPABILITY_STATES
    # Every state is exactly one of: evictable hint, owning, or the lone
    # unsupported terminal.
    assert (
        ALL_CAPABILITY_STATES
        == EVICTABLE_CAPABILITY_STATES | OWNING_CAPABILITY_STATES | {"unsupported"}
    )

    # Policy construction validates capability membership.
    route = _openai_route()
    with pytest.raises(ValueError):
        NativeCompactionPolicy(route=route, capability="bogus")
    # Terminal states cannot transition onward.
    terminal = NativeCompactionPolicy(route=route, capability="unsupported")
    with pytest.raises(ValueError):
        terminal.transition("shape_accepted")


def test_build_native_request_overrides():
    # The continuation gate rejects previous_response_id outright.
    with pytest.raises(ValueError, match="previous_response_id"):
        validate_responses_continuation_overrides({"previous_response_id": "resp_1"})
    # The native gate rejects every reserved protocol field.
    for reserved in NATIVE_RESPONSES_RESERVED_REQUEST_OVERRIDE_KEYS:
        with pytest.raises(ValueError, match="reserved"):
            validate_native_request_overrides({reserved: "x"})
    # Ordinary custom fields pass through untouched.
    custom = {"temperature": 0.5, "max_output_tokens": 128}
    assert validate_native_request_overrides(custom) == custom
    assert validate_native_request_overrides(None) == {}

    route = _openai_route()
    policy = NativeCompactionPolicy(route=route, capability="replay_verified")
    overrides = build_native_request_overrides(
        custom, mode="native", policy=policy, compact_threshold=50_000
    )
    assert overrides["temperature"] == 0.5
    # Native mode on a supported live route injects the context-management field.
    assert "context_management" in overrides
    # Non-native modes never inject the provider-owned field.
    hermes_overrides = build_native_request_overrides(
        custom, mode="hermes", policy=policy, compact_threshold=50_000
    )
    assert "context_management" not in hermes_overrides
    assert hermes_overrides == custom


def test_session_db_none_graceful():
    route = _openai_route()

    # Read paths short-circuit without a durable boundary.
    outcome = read_policy_for_route(None, None, route)
    assert outcome.outcome == "not_attempted"
    assert not outcome.failed_closed
    assert outcome.policy is not None and outcome.policy.capability == "unknown"
    assert read_policy_for_route(None, "session-1", route).outcome == "not_attempted"

    # Loaders fall back to fresh default custody instead of raising.
    assert load_policy_for_route(None, None, route).capability == "unknown"
    assert load_policy_for_route(None, "session-1", route).capability == "unknown"
    assert load_compaction_ledger(None, None).routes == {}
    assert load_compaction_ledger(None, "session-1").routes == {}

    # The CAS persistence entry point fails closed without a durable boundary.
    receipt = persist_policy_compare_and_set(
        None, None, NativeCompactionPolicy(route=route)
    )
    assert receipt.outcome == "failed_closed"
    assert receipt.failed_closed
    assert receipt.durable_revision is None
    assert receipt.error == "durable_session_boundary_missing"

    # The full agent-level decision path runs without a session db.
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5.1-codex",
        compression_enabled=True,
        codex_responses_auto_compaction="native",
        session_id=None,
    )
    assert should_defer_automatic_hermes_compaction(agent) is False
    assert agent._native_compaction_read_status.outcome == "not_attempted"
    assert agent._native_compaction_policy.capability == "unknown"


def test_route_for_request_issuer_detection():
    # Codex backend and xAI/github endpoints map to their canonical issuers.
    codex = route_for_request(
        provider="openai-codex",
        endpoint="https://user:secret@chatgpt.com/backend-api/codex/v1",
        model="gpt-5.1-codex",
    )
    assert codex.issuer_kind == "codex_backend"
    assert "secret" not in codex.endpoint
    assert codex.endpoint == "https://chatgpt.com/backend-api/codex/v1"

    xai = route_for_request(
        provider="xai", endpoint="https://api.x.ai/v1", model="grok-codex"
    )
    assert xai.issuer_kind == "xai_responses"

    github = route_for_request(
        provider="github-copilot",
        endpoint="https://api.githubcopilot.com/responses",
        model="gpt-5.1-codex",
    )
    assert github.issuer_kind == "github_responses"


# ---------------------------------------------------------------------------
# Adapter replay branch (agent/codex_responses_adapter.py)
# ---------------------------------------------------------------------------


def _openai_route_dict() -> dict:
    return {
        "issuer_kind": "openai",
        "endpoint": "https://chatgpt.com/backend-api/codex",
        "model": "gpt-5.1-codex",
    }


def _boundary_messages_with_sidecar(*, issuer: str = "openai") -> list:
    """Transcript whose boundary assistant message carries an ordered
    compaction sidecar stamped for *issuer* (same shape as what
    _normalize_codex_response persists for a compaction turn)."""
    route = _openai_route_dict() if issuer == "openai" else {
        "issuer_kind": issuer,
        "endpoint": "https://api.x.ai/v1",
        "model": "grok-4.5",
    }
    return [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {
            "role": "assistant",
            "content": "boundary visible text",
            "codex_output_items": [
                {
                    "type": "compaction",
                    "encrypted_content": "opaque-compact-state",
                    "_issuer_kind": issuer,
                    "_compaction_route": route,
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "boundary visible text"}],
                    "_issuer_kind": issuer,
                    "_compaction_route": route,
                },
            ],
        },
        {"role": "user", "content": "new user"},
    ]


def test_replay_compaction_sidecar_prefix():
    """A same-issuer compaction sidecar becomes the new request prefix: the
    ordered output items replace all earlier history in the API projection
    (compaction item first), while later transcript messages follow."""
    items = _chat_messages_to_responses_input(
        _boundary_messages_with_sidecar(),
        current_issuer_kind="openai",
        current_compaction_route=_openai_route_dict(),
    )
    assert [item.get("type") for item in items[:2]] == ["compaction", "message"]
    assert items[0] == {
        "type": "compaction",
        "encrypted_content": "opaque-compact-state",
    }
    assert items[1]["content"][0]["text"] == "boundary visible text"
    assert items[2] == {"role": "user", "content": "new user"}
    # The pre-compaction history is replaced, not replayed alongside it.
    assert all(item.get("content") != "old user" for item in items)


def test_replay_compaction_digest_mismatch_fallback():
    """A sidecar whose digest does not match the expected committed checkpoint
    fails open to an intact transcript replay — stale in-memory checkpoints
    must never win the projection."""
    messages = _boundary_messages_with_sidecar()
    bogus_digest = compaction_checkpoint_digest(
        [{"type": "compaction", "encrypted_content": "newer-uncommitted"}]
    )
    items = _chat_messages_to_responses_input(
        messages,
        current_issuer_kind="openai",
        current_compaction_route=_openai_route_dict(),
        expected_compaction_digest=bogus_digest,
    )
    assert all(item.get("type") != "compaction" for item in items)
    assert [item.get("content") for item in items] == [
        "old user",
        "old assistant",
        "boundary visible text",
        "new user",
    ]


def test_replay_compaction_foreign_issuer_ignored():
    """A sidecar stamped for a different issuer is not replayed — the
    encrypted blob is only decryptable by the minting endpoint."""
    items = _chat_messages_to_responses_input(
        _boundary_messages_with_sidecar(issuer="openai"),
        current_issuer_kind="xai",
        current_compaction_route={
            "issuer_kind": "xai",
            "endpoint": "https://api.x.ai/v1",
            "model": "grok-4.5",
        },
    )
    assert all(item.get("type") != "compaction" for item in items)
    assert [item.get("content") for item in items] == [
        "old user",
        "old assistant",
        "boundary visible text",
        "new user",
    ]


def test_preflight_accepts_user_summary_message():
    """After native compaction the provider may return a user-role summary
    message inside the output items; preflight must accept it as the first
    post-compaction input instead of rejecting every non-assistant message
    item."""
    normalized = _preflight_codex_input_items(
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "summarized context"}],
            }
        ]
    )
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"][0]["text"] == "summarized context"

    # String content is normalized to the Responses input_text part shape.
    normalized_str = _preflight_codex_input_items(
        [{"type": "message", "role": "user", "content": "plain summary"}]
    )
    assert normalized_str[0]["role"] == "user"
    assert normalized_str[0]["content"] == [
        {"type": "input_text", "text": "plain summary"}
    ]


# ---------------------------------------------------------------------------
# Local-summary bridge (agent/responses_compaction.py + context_compressor)
# ---------------------------------------------------------------------------


def _local_route() -> NativeCompactionRoute:
    """Route for a local DeepSeek-class fork (issuer falls to the catch-all
    ``other:<endpoint>`` bucket)."""
    return NativeCompactionRoute(
        issuer_kind="other:https://opencode.ai/zen/go/v1",
        endpoint="https://opencode.ai/zen/go/v1",
        model="deepseek-v4-flash",
    )


def _bridge_compressor(*, compression_remote: str = "auto") -> ContextCompressor:
    """ContextCompressor with mocked deps and a tight tail budget so the
    sidecar mount is observable and fast (no LLM/network calls)."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        c = ContextCompressor(
            model="deepseek-v4-flash",
            base_url="https://opencode.ai/zen/go/v1",
            provider="opencode-go",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
            compression_remote=compression_remote,
        )
        c.tail_token_budget = 10
        return c


def _bridge_transcript() -> list:
    """Compressible transcript whose head ends on a user turn, so the
    summary lands as a standalone assistant message (the ideal mount)."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial"},
    ] + [
        {"role": "user", "content": f"middle q{i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"middle reply {i}"}
        for i in range(12)
    ] + [
        {"role": "user", "content": "the visible question"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "follow up"},
    ]


def test_wrap_unwrap_roundtrip():
    route = _local_route()
    created_at = 1_752_000_000.5
    item = wrap_summary_as_compaction_item(
        "handoff summary body",
        route,
        created_at=created_at,
        token_savings_est=12_345,
    )
    assert item["type"] == "compaction"
    assert item["encrypted_content"].startswith(BRIDGE_ENVELOPE_PREFIX)
    payload = unwrap_compaction_item(item)
    assert payload["v"] == _BRIDGE_ENVELOPE_VERSION
    assert payload["summary_text"] == "handoff summary body"
    assert payload["created_at"] == created_at
    assert payload["token_savings_est"] == 12_345
    assert payload["route"] == route.to_dict()
    assert NativeCompactionRoute.from_dict(payload["route"]) == route

    # Malformed / non-bridge payloads are rejected, never silently decoded.
    with pytest.raises(ValueError, match="not a bridge"):
        unwrap_compaction_item(
            {"type": "compaction", "encrypted_content": "opaque-provider-blob"}
        )
    with pytest.raises(ValueError, match="encrypted_content"):
        unwrap_compaction_item({"type": "compaction"})


def test_wrap_truncates_oversized_summary():
    route = _local_route()
    huge = "x" * 40_000
    item = wrap_summary_as_compaction_item(huge, route, max_item_chars=2000)
    assert item["type"] == "compaction"
    envelope = item["encrypted_content"]
    # The envelope header survives the cap and the item stays on budget.
    assert envelope.startswith(BRIDGE_ENVELOPE_PREFIX)
    assert len(envelope) <= 2000
    payload = unwrap_compaction_item(item)
    assert payload["v"] == _BRIDGE_ENVELOPE_VERSION
    # Truncation keeps the summary TAIL — the newest recovered state lives at
    # the end of a handoff summary — never the head.
    assert payload["summary_text"].endswith("x" * 200)
    assert payload["summary_text"] == huge[-len(payload["summary_text"]) :]
    assert len(payload["summary_text"]) < len(huge)
    assert payload["route"] == route.to_dict()

    # A cap that cannot even hold the envelope header fails closed.
    with pytest.raises(ValueError, match="too small"):
        wrap_summary_as_compaction_item("tiny", route, max_item_chars=10)


def test_sidecar_carries_route_stamps():
    route = _local_route()
    item = wrap_summary_as_compaction_item("checkpoint body", route)
    sidecar = compaction_item_to_sidecar(item, route)
    assert sidecar["_issuer_kind"] == route.issuer_kind
    assert sidecar["_compaction_route"] == route.to_dict()
    # The opaque payload is untouched by the stamping.
    assert sidecar["type"] == "compaction"
    assert sidecar["encrypted_content"] == item["encrypted_content"]
    # The stamped sidecar satisfies the adapter replay fence verbatim:
    # same issuer, same canonical route dict.
    assert sidecar["_issuer_kind"] == route.issuer_kind
    assert sidecar["_compaction_route"] == route.to_dict()


def test_wrap_rejects_empty_summary():
    route = _local_route()
    with pytest.raises(ValueError, match="non-empty"):
        wrap_summary_as_compaction_item("", route)
    with pytest.raises(ValueError, match="non-empty"):
        wrap_summary_as_compaction_item("   \n\t ", route)
    with pytest.raises(ValueError, match="NativeCompactionRoute"):
        wrap_summary_as_compaction_item("summary", "not-a-route")  # type: ignore[arg-type]


def test_compress_attaches_sidecar_when_remote_enabled():
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY,
        SUMMARY_PREFIX,
    )

    c = _bridge_compressor(compression_remote="auto")
    mocked = f"{SUMMARY_PREFIX}\nrolled-up middle summary"
    with patch.object(c, "_generate_summary", return_value=mocked):
        result = c.compress(_bridge_transcript(), current_tokens=90_000)

    # Exactly one boundary message carries the sidecar, and it is the fresh
    # assistant-role summary carrier — the compaction response itself.
    carriers = [m for m in result if m.get("codex_output_items")]
    assert len(carriers) == 1
    boundary = carriers[0]
    assert boundary["role"] == "assistant"
    assert boundary.get(COMPRESSED_SUMMARY_METADATA_KEY)

    # The sidecar is the wrapped, route-stamped compaction item.
    sidecar = boundary["codex_output_items"]
    assert len(sidecar) == 1
    item = sidecar[0]
    expected_route = route_for_request(
        provider="opencode-go",
        endpoint="https://opencode.ai/zen/go/v1",
        model="deepseek-v4-flash",
    )
    assert item["type"] == "compaction"
    assert item["encrypted_content"].startswith(BRIDGE_ENVELOPE_PREFIX)
    assert item["_issuer_kind"] == expected_route.issuer_kind
    assert item["_compaction_route"] == expected_route.to_dict()
    payload = unwrap_compaction_item(item)
    assert payload["v"] == _BRIDGE_ENVELOPE_VERSION
    assert "rolled-up middle summary" in payload["summary_text"]
    assert payload["route"] == expected_route.to_dict()

    # End to end: the responses adapter accepts the mounted sidecar as the
    # new request prefix and the live tail survives the projection.
    items = _chat_messages_to_responses_input(
        result,
        current_issuer_kind=expected_route.issuer_kind,
        current_compaction_route=expected_route.to_dict(),
    )
    assert items[0]["type"] == "compaction"
    assert items[0]["encrypted_content"].startswith(BRIDGE_ENVELOPE_PREFIX)
    assert "the visible question" in str([i.get("content") for i in items])


def test_compress_unchanged_when_remote_off():
    from agent.context_compressor import SUMMARY_PREFIX

    mocked = f"{SUMMARY_PREFIX}\nrolled-up middle summary"
    c_off = _bridge_compressor(compression_remote="off")
    c_on = _bridge_compressor(compression_remote="on")
    messages = _bridge_transcript()
    with patch.object(c_off, "_generate_summary", return_value=mocked), patch.object(
        c_on, "_generate_summary", return_value=mocked
    ):
        result_off = c_off.compress(messages, current_tokens=90_000)
        result_on = c_on.compress(messages, current_tokens=90_000)
    # remote=off never mounts: output carries no sidecar at all.
    assert all("codex_output_items" not in m for m in result_off)
    # And it is byte-identical to the mounted output minus the sidecar key —
    # the only difference the bridge introduces is the sidecar itself.
    stripped_on = [
        {k: v for k, v in m.items() if k != "codex_output_items"}
        for m in result_on
    ]
    assert stripped_on == result_off
    # Unrecognized remote values normalize to the conservative off.
    assert _bridge_compressor(compression_remote="bogus").compression_remote == "off"


def test_compress_fallback_never_mounts_sidecar():
    """The deterministic fallback carries no recovered information worth
    checkpointing — even with remote enabled it must never mount."""
    c = _bridge_compressor(compression_remote="on")
    with patch.object(c, "_generate_summary", return_value=None):
        result = c.compress(_bridge_transcript(), current_tokens=90_000)
    assert c._last_summary_fallback_used is True
    assert all("codex_output_items" not in m for m in result)

