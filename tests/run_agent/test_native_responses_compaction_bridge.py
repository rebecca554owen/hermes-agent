"""Unit tests for the ported native Responses compaction policy primitives.

Covers the dependency-free policy/ledger layer in ``agent/responses_compaction.py``:
the capability state machine, route normalization, checkpoint digests, request
override validation, and the ``session_db=None`` graceful paths. The durable
``session_db``-backed persistence paths (``get_codex_responses_compaction_state`` /
``compare_and_set_codex_responses_compaction_state``) belong to a later task and
are not exercised here.
"""

from types import SimpleNamespace

import pytest

from agent.responses_compaction import (
    ALL_CAPABILITY_STATES,
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
    build_native_request_overrides,
    compaction_checkpoint_digest,
    compaction_route_key,
    load_compaction_ledger,
    load_policy_for_route,
    normalize_compaction_endpoint,
    persist_policy_compare_and_set,
    read_policy_for_route,
    route_for_request,
    should_defer_automatic_hermes_compaction,
    validate_native_request_overrides,
    validate_responses_continuation_overrides,
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
