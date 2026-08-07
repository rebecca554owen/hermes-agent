"""Policy and safety helpers for native OpenAI Responses compaction.

This module is dependency-free on purpose: request construction, persistence,
and the conversation loop can share one route-scoped state contract without
creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import logging
import re
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

COMPACTION_LEDGER_VERSION = 3
MAX_COMPACTION_LEDGER_ROUTES = 32
CAPABILITY_STATES = (
    "unknown",
    "shape_accepted",
    "item_observed",
    "replay_verified",
)
TERMINAL_STATES = {"unsupported", "quarantined"}
ALL_CAPABILITY_STATES = set(CAPABILITY_STATES) | TERMINAL_STATES
logger = logging.getLogger(__name__)
_ROUTE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_INVALID_ENDPOINT_SENTINEL = "invalid://redacted"
OWNING_CAPABILITY_STATES = {"item_observed", "replay_verified", "quarantined"}
CHECKPOINT_REQUIRED_CAPABILITY_STATES = {"item_observed", "replay_verified"}
EVICTABLE_CAPABILITY_STATES = {"unknown", "shape_accepted"}
_MANUAL_HERMES_AUTHORIZATION_ATTR = "_manual_hermes_compaction_authorization"
_EMERGENCY_HERMES_AUTHORIZATION_ATTR = "_emergency_hermes_compaction_authorization"
NATIVE_RESPONSES_RESERVED_REQUEST_OVERRIDE_KEYS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "reasoning",
        "include",
        "store",
        "previous_response_id",
        "context_management",
    }
)


class NativeCompactionStateError(ValueError):
    """Persisted native-compaction state is malformed or non-canonical."""


class NativeCompactionReadError(RuntimeError):
    """Durable native-compaction custody could not be read safely."""


def validate_responses_continuation_overrides(
    request_overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Copy overrides while prohibiting hidden provider-side continuation."""
    if request_overrides is None:
        return {}
    if not isinstance(request_overrides, Mapping):
        raise ValueError("Responses request overrides must be an object.")
    if "previous_response_id" in request_overrides:
        raise ValueError(
            "previous_response_id is prohibited as a reserved native Responses "
            "field; history must remain bound to the durable local transcript."
        )
    return dict(request_overrides)


def validate_native_request_overrides(
    request_overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Copy safe provider overrides or reject protocol-field collisions."""
    overrides = validate_responses_continuation_overrides(request_overrides)
    collisions = sorted(
        NATIVE_RESPONSES_RESERVED_REQUEST_OVERRIDE_KEYS.intersection(overrides)
    )
    if collisions:
        raise ValueError(
            "Native Responses request override contains reserved native "
            f"Responses field(s): {', '.join(collisions)}."
        )
    return overrides


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NativeCompactionStateError(f"{field} must be a non-negative integer")
    return value


def _strict_optional_text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 512:
        raise NativeCompactionStateError(
            f"{field} must be null or a non-empty string of at most 512 characters"
        )
    return value


def normalize_compaction_endpoint(value: str) -> str:
    """Return a stable, credential-free route identity for capability state."""
    raw = (value or "").strip()
    if not raw:
        return ""
    # Remove suffixes before attempting any parse. ``urlsplit`` treats values
    # such as ``user:secret@example/v1`` as a custom scheme with no authority,
    # so returning its original string on parse failure would persist the
    # credentials and query verbatim.
    without_fragment = raw.split("#", 1)[0]
    without_suffix = without_fragment.split("?", 1)[0].strip()
    if (
        not without_suffix
        or any(ord(char) < 32 for char in without_suffix)
        or any(char.isspace() for char in without_suffix)
        or "\\" in without_suffix
    ):
        return _INVALID_ENDPOINT_SENTINEL

    def _credential_free_authority(parsed: Any) -> Optional[str]:
        try:
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return None
        if not isinstance(hostname, str) or not hostname:
            return None
        host = hostname.lower()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"{host}:{port}" if port is not None else host

    absolute = urlsplit(without_suffix)
    if absolute.scheme and absolute.netloc:
        authority = _credential_free_authority(absolute)
        if authority is None:
            return _INVALID_ENDPOINT_SENTINEL
        return urlunsplit(
            (
                absolute.scheme.lower(),
                authority,
                absolute.path.rstrip("/"),
                "",
                "",
            )
        )

    # Parse scheme-less host/path values as network-path references. This
    # recovers a useful ``//host/path`` identity while stripping userinfo.
    candidate = without_suffix[2:] if without_suffix.startswith("//") else without_suffix
    network_path = urlsplit(f"//{candidate}")
    authority = _credential_free_authority(network_path)
    first_segment = candidate.split("/", 1)[0]
    looks_like_authority = bool(
        authority
        and (
            "@" in first_segment
            or "." in authority
            or ":" in authority
            or authority == "localhost"
        )
    )
    if looks_like_authority:
        return urlunsplit(
            ("", authority or "", network_path.path.rstrip("/"), "", "")
        )

    # Relative/custom endpoints have no authority to key. Preserve only their
    # credential-free path when possible; otherwise use a stable sentinel. An
    # ``@`` anywhere in this fallback is treated as userinfo and removed.
    relative = without_suffix.rsplit("@", 1)[-1].rstrip("/")
    if not relative or any(marker in relative for marker in ("?", "#", "@")):
        return _INVALID_ENDPOINT_SENTINEL
    return relative


@dataclass(frozen=True)
class NativeCompactionRoute:
    issuer_kind: str
    endpoint: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "issuer_kind", (self.issuer_kind or "").strip().lower())
        object.__setattr__(self, "endpoint", normalize_compaction_endpoint(self.endpoint))
        object.__setattr__(self, "model", (self.model or "").strip())

    def to_dict(self) -> Dict[str, str]:
        return {
            "issuer_kind": self.issuer_kind,
            "endpoint": self.endpoint,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeCompactionRoute":
        if not isinstance(value, Mapping):
            raise NativeCompactionStateError("route must be an object")
        expected = {"issuer_kind", "endpoint", "model"}
        if set(value) != expected:
            raise NativeCompactionStateError("route has missing or unexpected fields")
        if any(not isinstance(value[key], str) for key in expected):
            raise NativeCompactionStateError("route fields must be strings")
        route = cls(
            issuer_kind=value["issuer_kind"],
            endpoint=value["endpoint"],
            model=value["model"],
        )
        if not route.issuer_kind or not route.endpoint or not route.model:
            raise NativeCompactionStateError("route fields must be non-empty")
        if route.to_dict() != dict(value):
            raise NativeCompactionStateError("route must use its canonical normalized form")
        return route


def compaction_route_key(route: NativeCompactionRoute) -> str:
    canonical = json.dumps(
        route.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeCompactionPolicy:
    route: NativeCompactionRoute
    capability: str = "unknown"
    revision: int = 0
    compaction_count: int = 0
    fallback_count: int = 0
    last_compaction_digest: Optional[str] = None
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.capability not in ALL_CAPABILITY_STATES:
            raise ValueError(f"invalid native compaction capability: {self.capability!r}")
        if self.revision < 0 or self.compaction_count < 0 or self.fallback_count < 0:
            raise ValueError("native compaction counters must be non-negative")

    def transition(self, target: str, *, error: Optional[str] = None) -> "NativeCompactionPolicy":
        if target not in ALL_CAPABILITY_STATES:
            raise ValueError(f"invalid native compaction capability: {target!r}")
        if target == self.capability:
            return replace(self, last_error=error if error is not None else self.last_error)
        if self.capability in TERMINAL_STATES:
            raise ValueError(f"terminal native compaction state {self.capability!r} cannot transition")
        if target in TERMINAL_STATES:
            return replace(self, capability=target, last_error=error)
        current_index = CAPABILITY_STATES.index(self.capability)
        target_index = CAPABILITY_STATES.index(target)
        if target_index <= current_index:
            raise ValueError(
                f"native compaction capability cannot regress from {self.capability!r} to {target!r}"
            )
        return replace(self, capability=target, last_error=error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "capability": self.capability,
            "revision": self.revision,
            "compaction_count": self.compaction_count,
            "fallback_count": self.fallback_count,
            "last_compaction_digest": self.last_compaction_digest,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeCompactionPolicy":
        if not isinstance(value, Mapping):
            raise NativeCompactionStateError("policy must be an object")
        expected = {
            "route",
            "capability",
            "revision",
            "compaction_count",
            "fallback_count",
            "last_compaction_digest",
            "last_error",
        }
        if set(value) != expected:
            raise NativeCompactionStateError("policy has missing or unexpected fields")
        capability = value["capability"]
        if not isinstance(capability, str) or capability not in ALL_CAPABILITY_STATES:
            raise NativeCompactionStateError("policy capability is invalid")
        digest = value["last_compaction_digest"]
        if digest is not None and (
            not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None
        ):
            raise NativeCompactionStateError(
                "last_compaction_digest must be null or a lowercase SHA-256 digest"
            )
        compaction_count = _strict_nonnegative_int(
            value["compaction_count"], "compaction_count"
        )
        if (compaction_count == 0) != (digest is None):
            raise NativeCompactionStateError(
                "compaction_count and last_compaction_digest are inconsistent"
            )
        if capability in {"item_observed", "replay_verified"} and (
            compaction_count == 0 or digest is None
        ):
            raise NativeCompactionStateError(
                "observed native compaction capability requires a checkpoint digest"
            )
        if capability in {"unknown", "shape_accepted"} and compaction_count:
            raise NativeCompactionStateError(
                "unobserved native compaction capability cannot carry checkpoint state"
            )
        return cls(
            route=NativeCompactionRoute.from_dict(value["route"]),
            capability=capability,
            revision=_strict_nonnegative_int(value["revision"], "revision"),
            compaction_count=compaction_count,
            fallback_count=_strict_nonnegative_int(
                value["fallback_count"], "fallback_count"
            ),
            last_compaction_digest=digest,
            last_error=_strict_optional_text(value["last_error"], "last_error"),
        )

    def to_ledger_entry(self) -> Dict[str, Any]:
        value = self.to_dict()
        value.pop("revision")
        return value

    @classmethod
    def from_ledger_entry(
        cls, value: Mapping[str, Any], *, revision: int
    ) -> "NativeCompactionPolicy":
        if not isinstance(value, Mapping):
            raise NativeCompactionStateError("ledger route entry must be an object")
        full = dict(value)
        full["revision"] = revision
        return cls.from_dict(full)


@dataclass(frozen=True)
class HermesCompactionAuthorization:
    """One-shot proof that an explicit manual rewrite owns this exact route."""

    route: NativeCompactionRoute
    session_id: Optional[str]
    policy_revision: int
    mode: str
    reason: str


@dataclass(frozen=True)
class EmergencyHermesCompactionAuthorization:
    """One-shot proof for a genuine provider-overflow Hermes rewrite."""

    route: NativeCompactionRoute
    session_id: Optional[str]
    policy_revision: int
    mode: str
    reason: str


@dataclass(frozen=True)
class NativeCompactionReadOutcome:
    """Explicit result of reading route custody from the durable ledger."""

    outcome: str
    policy: Optional[NativeCompactionPolicy]
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in {"succeeded", "not_attempted", "failed_closed"}:
            raise ValueError(f"invalid native compaction read outcome: {self.outcome!r}")
        if self.outcome == "failed_closed":
            if self.policy is not None or not self.error:
                raise ValueError(
                    "failed-closed native compaction read cannot carry policy custody"
                )
        elif self.policy is None or self.error is not None:
            raise ValueError(
                "successful or unattempted native compaction read requires a policy"
            )

    @property
    def succeeded(self) -> bool:
        return self.outcome == "succeeded"

    @property
    def failed_closed(self) -> bool:
        return self.outcome == "failed_closed"


@dataclass(frozen=True)
class NativeCompactionLedger:
    """Versioned, route-scoped durable capability ledger."""

    revision: int
    routes: Mapping[str, NativeCompactionPolicy]
    version: int = COMPACTION_LEDGER_VERSION

    def __post_init__(self) -> None:
        if self.version != COMPACTION_LEDGER_VERSION:
            raise NativeCompactionStateError(
                f"unsupported native compaction ledger version: {self.version!r}"
            )
        _strict_nonnegative_int(self.revision, "revision")
        if not isinstance(self.routes, Mapping):
            raise NativeCompactionStateError("routes must be an object")
        if len(self.routes) > MAX_COMPACTION_LEDGER_ROUTES:
            raise NativeCompactionStateError("native compaction route ledger is too large")
        normalized: Dict[str, NativeCompactionPolicy] = {}
        for key, policy in self.routes.items():
            if not isinstance(key, str) or _ROUTE_KEY_RE.fullmatch(key) is None:
                raise NativeCompactionStateError("route ledger key is invalid")
            if not isinstance(policy, NativeCompactionPolicy):
                raise NativeCompactionStateError("route ledger value is not a policy")
            if key != compaction_route_key(policy.route):
                raise NativeCompactionStateError("route ledger key does not match route")
            if policy.revision != self.revision:
                policy = replace(policy, revision=self.revision)
            normalized[key] = policy
        object.__setattr__(self, "routes", normalized)

    @classmethod
    def empty(cls) -> "NativeCompactionLedger":
        return cls(revision=0, routes={})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "revision": self.revision,
            "routes": {
                key: self.routes[key].to_ledger_entry()
                for key in sorted(self.routes)
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeCompactionLedger":
        if not isinstance(value, Mapping):
            raise NativeCompactionStateError("native compaction ledger must be an object")
        if set(value) != {"version", "revision", "routes"}:
            raise NativeCompactionStateError(
                "native compaction ledger has missing or unexpected fields"
            )
        version = value["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise NativeCompactionStateError("ledger version must be an integer")
        revision = _strict_nonnegative_int(value["revision"], "revision")
        raw_routes = value["routes"]
        if not isinstance(raw_routes, Mapping):
            raise NativeCompactionStateError("routes must be an object")
        routes: Dict[str, NativeCompactionPolicy] = {}
        for key, entry in raw_routes.items():
            if not isinstance(key, str) or _ROUTE_KEY_RE.fullmatch(key) is None:
                raise NativeCompactionStateError("route ledger key is invalid")
            policy = NativeCompactionPolicy.from_ledger_entry(
                entry, revision=revision
            )
            if key != compaction_route_key(policy.route):
                raise NativeCompactionStateError("route ledger key does not match route")
            routes[key] = policy
        return cls(revision=revision, routes=routes, version=version)

    def policy_for(self, route: NativeCompactionRoute) -> NativeCompactionPolicy:
        policy = self.routes.get(compaction_route_key(route))
        if policy is None:
            return NativeCompactionPolicy(route=route, revision=self.revision)
        return replace(policy, revision=self.revision)

    def with_policy(
        self, policy: NativeCompactionPolicy
    ) -> "NativeCompactionLedger":
        routes = dict(self.routes)
        key = compaction_route_key(policy.route)
        routes[key] = replace(policy, revision=self.revision)
        if len(routes) > MAX_COMPACTION_LEDGER_ROUTES:
            # Preserve terminal history and every ownership-bearing route. Only
            # non-owning discovery hints may be evicted under bounded capacity.
            for stale_key, stale_policy in list(routes.items()):
                if (
                    stale_key != key
                    and stale_policy.capability in EVICTABLE_CAPABILITY_STATES
                ):
                    routes.pop(stale_key)
                    break
            else:
                raise NativeCompactionStateError(
                    "native compaction route ledger is full of durable custody or terminal history"
                )
        return NativeCompactionLedger(revision=self.revision, routes=routes)


@dataclass(frozen=True)
class NativeCompactionTransitionReceipt:
    """Typed proof of one durable native-compaction state transition."""

    outcome: str
    policy: NativeCompactionPolicy
    ledger: NativeCompactionLedger
    durable_revision: Optional[int]
    attempts: int
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in {
            "committed",
            "conflict_reconciled",
            "failed_closed",
        }:
            raise ValueError(f"invalid native compaction receipt: {self.outcome!r}")
        if self.outcome == "failed_closed":
            if self.durable_revision is not None:
                raise ValueError("failed-closed receipt cannot claim a durable revision")
        elif self.durable_revision is None:
            raise ValueError("durable transition receipt requires a revision")

    @property
    def committed(self) -> bool:
        return self.outcome == "committed"

    @property
    def conflict_reconciled(self) -> bool:
        return self.outcome == "conflict_reconciled"

    @property
    def failed_closed(self) -> bool:
        return self.outcome == "failed_closed"

    @property
    def authorizes_transition(self) -> bool:
        return self.outcome in {"committed", "conflict_reconciled"}


def _structured_error(error: BaseException) -> tuple[Optional[int], Optional[Mapping[str, Any]]]:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None

    body = getattr(error, "body", None)
    if body is None and response is not None:
        try:
            body = response.json()
        except Exception:
            body = None
    if not isinstance(body, Mapping):
        return status_int, None
    nested = body.get("error")
    return status_int, nested if isinstance(nested, Mapping) else body


def is_structured_compaction_unsupported_error(error: BaseException) -> bool:
    """Classify only explicit compaction-parameter 400/422 rejections.

    Free-form message matching is intentionally forbidden: auth, rate-limit,
    timeout, generic invalid-request, and provider failures must not permanently
    downgrade native compaction.
    """
    status, detail = _structured_error(error)
    if status not in {400, 422} or detail is None:
        return False
    param = str(detail.get("param") or "").strip().lower()
    if not param:
        return False
    return "context_management" in param or "compact_threshold" in param


def policy_after_structured_compaction_rejection(
    policy: NativeCompactionPolicy,
) -> NativeCompactionPolicy:
    """Record provider rejection without releasing established route custody."""
    target = (
        "quarantined"
        if policy.capability in OWNING_CAPABILITY_STATES
        else "unsupported"
    )
    return policy.transition(
        target,
        error="provider_rejected_context_management",
    )


def context_management_for_threshold(threshold: int) -> list[dict[str, Any]]:
    value = int(threshold)
    if isinstance(threshold, bool) or value <= 0:
        raise ValueError("compact threshold must be a positive integer")
    return [{"type": "compaction", "compact_threshold": value}]


def _route_supports_native_compaction(route: NativeCompactionRoute) -> bool:
    if route.issuer_kind == "codex_backend":
        return True
    parsed = urlsplit(route.endpoint)
    return parsed.netloc.lower() == "api.openai.com"


def resolve_native_compaction_threshold(
    configured_threshold: int,
    *,
    hermes_threshold: int,
    safety_margin: int = 8_192,
) -> int:
    """Resolve a native threshold that fires before Hermes' fallback."""
    configured = int(configured_threshold)
    hermes = int(hermes_threshold)
    if isinstance(configured_threshold, bool) or configured <= 0:
        raise ValueError("native compact threshold must be a positive integer")
    if hermes <= 0:
        raise ValueError("Hermes compact threshold must be positive")
    if hermes > safety_margin:
        upper = hermes - safety_margin
    else:
        upper = max(1, int(hermes * 0.8))
    return min(configured, upper)


def build_native_request_overrides(
    request_overrides: Optional[Mapping[str, Any]],
    *,
    mode: str,
    policy: NativeCompactionPolicy,
    compact_threshold: int,
) -> Dict[str, Any]:
    """Add the native context-management field only on an allowed live route."""
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "native":
        overrides = validate_native_request_overrides(request_overrides)
    else:
        overrides = validate_responses_continuation_overrides(request_overrides)
    if (
        normalized_mode != "native"
        or policy.capability in TERMINAL_STATES
        or not _route_supports_native_compaction(policy.route)
    ):
        overrides.pop("context_management", None)
        return overrides
    overrides["context_management"] = context_management_for_threshold(
        compact_threshold
    )
    return overrides


def _has_compaction_item(items: Any) -> bool:
    return isinstance(items, list) and any(
        isinstance(item, Mapping)
        and item.get("type") == "compaction"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item.get("encrypted_content"))
        for item in items
    )


def has_compaction_item(items: Any) -> bool:
    """Public strict predicate used by persistence and continuation gates."""
    return _has_compaction_item(items)


def compaction_checkpoint_digest(items: Any) -> Optional[str]:
    """Digest the exact ordered compaction output checkpoint."""
    if not _has_compaction_item(items):
        return None
    try:
        canonical = json.dumps(
            items,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_compaction_sidecar(
    items: Any,
) -> tuple[NativeCompactionRoute, Optional[str]]:
    """Validate one exact ordered Responses checkpoint sidecar.

    Output item payloads remain opaque apart from the minimum fencing fields
    needed for custody: every item is an object with a type, every item carries
    one canonical route/issuer stamp, and compaction items carry non-empty
    encrypted content. Unknown provider fields are preserved.
    """
    if not isinstance(items, list) or not items:
        raise NativeCompactionStateError(
            "codex_output_items must be a non-empty ordered list"
        )
    route: Optional[NativeCompactionRoute] = None
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise NativeCompactionStateError(
                f"codex_output_items[{index}] must be an object"
            )
        item_type = item.get("type")
        if not isinstance(item_type, str) or not item_type:
            raise NativeCompactionStateError(
                f"codex_output_items[{index}].type must be a non-empty string"
            )
        issuer = item.get("_issuer_kind")
        route_value = item.get("_compaction_route")
        if not isinstance(issuer, str) or not issuer:
            raise NativeCompactionStateError(
                f"codex_output_items[{index}] is missing its issuer fence"
            )
        item_route = NativeCompactionRoute.from_dict(route_value)
        if issuer != item_route.issuer_kind:
            raise NativeCompactionStateError(
                f"codex_output_items[{index}] issuer does not match its route"
            )
        if route is None:
            route = item_route
        elif item_route != route:
            raise NativeCompactionStateError(
                "codex_output_items mixes multiple compaction routes"
            )
        if item_type == "compaction" and (
            not isinstance(item.get("encrypted_content"), str)
            or not item.get("encrypted_content")
        ):
            raise NativeCompactionStateError(
                f"codex_output_items[{index}] has invalid compaction content"
            )
    try:
        json.dumps(
            items,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NativeCompactionStateError(
            "codex_output_items must be canonical JSON data"
        ) from exc
    assert route is not None
    return route, compaction_checkpoint_digest(items)


def validate_compaction_message_sidecar(
    message: Any,
    *,
    message_index: Optional[int] = None,
) -> Optional[tuple[list[Any], NativeCompactionRoute, Optional[str]]]:
    """Validate a sidecar together with its explicit transcript custody."""
    location = (
        f"compaction transcript message {message_index}"
        if message_index is not None
        else "compaction transcript message"
    )
    if not isinstance(message, Mapping):
        raise NativeCompactionStateError(f"{location} must be an object")
    if "codex_output_items" not in message:
        return None
    items = message.get("codex_output_items")
    # Canonical SessionDB/export rows include nullable columns for every
    # message. SQL NULL is absence, not a malformed checkpoint.
    if items is None:
        return None
    if message.get("role") != "assistant":
        raise NativeCompactionStateError(
            f"{location} with codex_output_items must have explicit assistant role"
        )
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (json.JSONDecodeError, TypeError) as exc:
            raise NativeCompactionStateError(
                f"{location} has malformed sidecar JSON"
            ) from exc
    route, digest = validate_compaction_sidecar(items)
    return items, route, digest


def _checkpoint_index(messages: Any) -> set[tuple[str, str]]:
    if not isinstance(messages, list):
        raise NativeCompactionStateError("compaction transcript must be a list")
    checkpoints: set[tuple[str, str]] = set()
    for index, message in enumerate(messages):
        validated = validate_compaction_message_sidecar(
            message, message_index=index
        )
        if validated is None:
            continue
        _, route, digest = validated
        if digest is not None:
            checkpoints.add((compaction_route_key(route), digest))
    return checkpoints


def validate_compaction_lifecycle(
    ledger: NativeCompactionLedger,
    messages: Any,
) -> NativeCompactionLedger:
    """Validate the ledger and canonical transcript as one semantic value."""
    checkpoints = _checkpoint_index(messages)
    checkpoint_routes = {route_key for route_key, _digest in checkpoints}
    for route_key in checkpoint_routes:
        policy = ledger.routes.get(route_key)
        if policy is None or policy.capability not in OWNING_CAPABILITY_STATES:
            raise NativeCompactionStateError(
                "native compaction checkpoint is missing owning route custody"
            )
    for route_key, policy in ledger.routes.items():
        if policy.capability not in CHECKPOINT_REQUIRED_CAPABILITY_STATES:
            continue
        digest = policy.last_compaction_digest
        if not digest or (route_key, digest) not in checkpoints:
            raise NativeCompactionStateError(
                "native compaction owner is missing its matching checkpoint"
            )
    return ledger


def derive_compaction_lifecycle(
    ledger: NativeCompactionLedger,
    messages: Any,
    *,
    quarantine_error: str,
) -> NativeCompactionLedger:
    """Return a publication-safe ledger for a rewritten child/transcript.

    Strict sidecar shape errors are rejected. Valid ledgers whose owning route
    lost its exact checkpoint are durably transitioned to quarantine before
    the new transcript is published.
    """
    checkpoints = _checkpoint_index(messages)
    routes = dict(ledger.routes)
    changed = False
    for route_key, policy in list(routes.items()):
        if policy.capability not in CHECKPOINT_REQUIRED_CAPABILITY_STATES:
            continue
        digest = policy.last_compaction_digest
        if digest and (route_key, digest) in checkpoints:
            continue
        routes[route_key] = replace(
            policy,
            capability="quarantined",
            last_error=quarantine_error,
        )
        changed = True
    if not changed:
        return ledger
    return NativeCompactionLedger(revision=ledger.revision + 1, routes=routes)


def advance_policy_after_success(
    policy: NativeCompactionPolicy,
    *,
    codex_output_items: Any,
    replay_attempted: bool,
) -> NativeCompactionPolicy:
    """Advance capability only from observable successful wire behavior."""
    if policy.capability in TERMINAL_STATES:
        return policy
    updated = policy
    saw_compaction = _has_compaction_item(codex_output_items)
    if saw_compaction:
        if updated.capability in {"unknown", "shape_accepted"}:
            updated = updated.transition("item_observed")
        digest = compaction_checkpoint_digest(codex_output_items)
        if digest and digest != updated.last_compaction_digest:
            updated = replace(
                updated,
                compaction_count=updated.compaction_count + 1,
                last_compaction_digest=digest,
            )
    elif updated.capability == "unknown":
        updated = updated.transition("shape_accepted")
    if replay_attempted and updated.capability == "item_observed":
        updated = updated.transition("replay_verified")
    return updated


def should_defer_hermes_compaction(
    mode: str,
    policy: NativeCompactionPolicy,
) -> bool:
    """Return whether Hermes automatic compaction should be skipped.

    ``native`` requires observed provider state; explicit ``off`` is an
    evaluation-only request to disable automatic compaction entirely.
    """
    mode_value = str(mode or "").strip().lower()
    if mode_value == "off":
        return True
    return (
        mode_value in {"native", "hermes"}
        and _route_supports_native_compaction(policy.route)
        and policy.capability in OWNING_CAPABILITY_STATES
    )


def route_for_request(
    *, provider: str, endpoint: str, model: str
) -> NativeCompactionRoute:
    normalized_endpoint = normalize_compaction_endpoint(endpoint)
    parsed = urlsplit(normalized_endpoint)
    provider_name = str(provider or "").strip().lower()
    if provider_name == "openai-codex" or (
        parsed.netloc.lower() == "chatgpt.com"
        and "/backend-api/codex" in parsed.path.lower()
    ):
        issuer = "codex_backend"
    elif provider_name in {"xai", "xai-oauth"} or parsed.netloc.lower() == "api.x.ai":
        issuer = "xai_responses"
    elif "githubcopilot.com" in parsed.netloc.lower() or parsed.netloc.lower() == "models.github.ai":
        issuer = "github_responses"
    else:
        issuer = f"other:{normalized_endpoint}"
    return NativeCompactionRoute(
        issuer_kind=issuer,
        endpoint=normalized_endpoint,
        model=model,
    )


def read_policy_for_route(
    session_db: Any,
    session_id: Optional[str],
    route: NativeCompactionRoute,
) -> NativeCompactionReadOutcome:
    if session_db is None or not session_id:
        return NativeCompactionReadOutcome(
            outcome="not_attempted",
            policy=NativeCompactionPolicy(route=route),
        )
    try:
        raw = session_db.get_codex_responses_compaction_state(session_id)
        ledger = NativeCompactionLedger.from_dict(raw)
    except Exception as exc:
        logger.error(
            "Durable native Responses compaction custody is unreadable; "
            "failing closed without changing route state: %s",
            exc,
        )
        return NativeCompactionReadOutcome(
            outcome="failed_closed",
            policy=None,
            error="invalid_durable_compaction_state",
        )
    return NativeCompactionReadOutcome(
        outcome="succeeded",
        policy=ledger.policy_for(route),
    )


def load_policy_for_route(
    session_db: Any,
    session_id: Optional[str],
    route: NativeCompactionRoute,
) -> NativeCompactionPolicy:
    """Load durable route custody or raise instead of inventing a transition."""
    outcome = read_policy_for_route(session_db, session_id, route)
    if outcome.failed_closed:
        raise NativeCompactionReadError(
            outcome.error or "native compaction durable read failed"
        )
    assert outcome.policy is not None
    return outcome.policy


def load_compaction_ledger(
    session_db: Any, session_id: Optional[str]
) -> NativeCompactionLedger:
    if session_db is None or not session_id:
        return NativeCompactionLedger.empty()
    raw = session_db.get_codex_responses_compaction_state(session_id)
    return NativeCompactionLedger.from_dict(raw)


def _merge_policy(
    current: NativeCompactionPolicy,
    desired: NativeCompactionPolicy,
) -> NativeCompactionPolicy:
    if current.route != desired.route:
        return replace(desired, revision=current.revision)
    if current.capability in TERMINAL_STATES:
        capability = current.capability
        last_error = current.last_error
    elif desired.capability in TERMINAL_STATES:
        capability = desired.capability
        last_error = desired.last_error
    else:
        capability = max(
            (current.capability, desired.capability),
            key=CAPABILITY_STATES.index,
        )
        last_error = desired.last_error or current.last_error
    return NativeCompactionPolicy(
        route=current.route,
        capability=capability,
        revision=current.revision,
        compaction_count=max(current.compaction_count, desired.compaction_count),
        fallback_count=max(current.fallback_count, desired.fallback_count),
        last_compaction_digest=(
            desired.last_compaction_digest
            if desired.compaction_count >= current.compaction_count
            else current.last_compaction_digest
        ),
        last_error=last_error,
    )


def failed_closed_transition_receipt(
    desired: NativeCompactionPolicy,
    *,
    error: str,
    ledger: Optional[NativeCompactionLedger] = None,
    attempts: int = 0,
) -> NativeCompactionTransitionReceipt:
    """Build the sole non-durable receipt shape and quarantine in memory."""
    effective_ledger = ledger or NativeCompactionLedger.empty()
    safe_policy = replace(
        desired,
        capability="quarantined",
        last_error=error,
        revision=effective_ledger.revision,
    )
    routes = dict(effective_ledger.routes)
    key = compaction_route_key(safe_policy.route)
    if key in routes or len(routes) < MAX_COMPACTION_LEDGER_ROUTES:
        routes[key] = safe_policy
        effective_ledger = NativeCompactionLedger(
            revision=effective_ledger.revision,
            routes=routes,
        )
    return NativeCompactionTransitionReceipt(
        outcome="failed_closed",
        policy=safe_policy,
        ledger=effective_ledger,
        durable_revision=None,
        attempts=attempts,
        error=error,
    )


_FAILED_RECEIPTS_ATTR = "_native_compaction_failed_receipts_by_route"
_FAILED_RECEIPTS_OVERFLOW_ATTR = "_native_compaction_failed_receipts_overflow"


def _failed_receipt_registry(
    agent: Any, *, create: bool
) -> Dict[str, NativeCompactionTransitionReceipt]:
    registry = getattr(agent, _FAILED_RECEIPTS_ATTR, None)
    if isinstance(registry, dict):
        valid = len(registry) <= MAX_COMPACTION_LEDGER_ROUTES and all(
            isinstance(key, str)
            and _ROUTE_KEY_RE.fullmatch(key) is not None
            and isinstance(receipt, NativeCompactionTransitionReceipt)
            and receipt.failed_closed
            and key == compaction_route_key(receipt.policy.route)
            for key, receipt in registry.items()
        )
        if valid:
            return registry
        setattr(agent, _FAILED_RECEIPTS_OVERFLOW_ATTR, True)
        registry = {}
        if create:
            setattr(agent, _FAILED_RECEIPTS_ATTR, registry)
        return registry
    if registry is not None:
        setattr(agent, _FAILED_RECEIPTS_OVERFLOW_ATTR, True)
    if not create:
        return {}
    registry = {}
    setattr(agent, _FAILED_RECEIPTS_ATTR, registry)
    return registry


def record_native_compaction_transition_receipt(
    agent: Any, receipt: NativeCompactionTransitionReceipt
) -> None:
    """Publish the latest receipt and retain unresolved failure per route."""
    agent._native_compaction_transition_receipt = receipt
    registry = _failed_receipt_registry(agent, create=True)
    key = compaction_route_key(receipt.policy.route)
    if receipt.failed_closed:
        if key not in registry and len(registry) >= MAX_COMPACTION_LEDGER_ROUTES:
            setattr(agent, _FAILED_RECEIPTS_OVERFLOW_ATTR, True)
            return
        registry[key] = receipt
    elif receipt.authorizes_transition:
        registry.pop(key, None)


def persist_policy_compare_and_set(
    session_db: Any,
    session_id: Optional[str],
    desired: NativeCompactionPolicy,
) -> NativeCompactionTransitionReceipt:
    """Merge one route into the v3 ledger with one bounded CAS retry."""
    if session_db is None or not session_id:
        return failed_closed_transition_receipt(
            desired,
            error="durable_session_boundary_missing",
        )
    last_ledger: Optional[NativeCompactionLedger] = None
    for attempt in range(2):
        try:
            ledger = load_compaction_ledger(session_db, session_id)
            last_ledger = ledger
            current = ledger.policy_for(desired.route)
            merged = _merge_policy(current, desired)
            next_ledger = ledger.with_policy(merged)
            if session_db.compare_and_set_codex_responses_compaction_state(
                session_id,
                expected_revision=ledger.revision,
                state=next_ledger.to_dict(),
            ):
                committed_ledger = NativeCompactionLedger(
                    revision=ledger.revision + 1,
                    routes=next_ledger.routes,
                )
                committed_policy = committed_ledger.policy_for(desired.route)
                outcome = "committed" if attempt == 0 else "conflict_reconciled"
                return NativeCompactionTransitionReceipt(
                    outcome=outcome,
                    policy=committed_policy,
                    ledger=committed_ledger,
                    durable_revision=committed_ledger.revision,
                    attempts=attempt + 1,
                )
        except Exception as exc:
            logger.error(
                "Failed to persist native Responses compaction ledger; "
                "quarantining in memory: %s",
                exc,
            )
            return failed_closed_transition_receipt(
                desired,
                error="capability_state_persistence_failed",
                ledger=last_ledger,
                attempts=attempt + 1,
            )
    logger.error(
        "Native Responses compaction ledger CAS remained contended; "
        "quarantining route in memory"
    )
    return failed_closed_transition_receipt(
        desired,
        error="capability_state_persistence_conflict",
        ledger=last_ledger,
        attempts=2,
    )


def resolve_native_compaction_receipt_for_durable_read(
    agent: Any,
    read_outcome: NativeCompactionReadOutcome,
) -> None:
    """Resolve only the route proven by a successful durable ledger read."""
    if not read_outcome.succeeded or read_outcome.policy is None:
        return
    route = read_outcome.policy.route
    registry = _failed_receipt_registry(agent, create=False)
    registry.pop(compaction_route_key(route), None)
    latest = getattr(agent, "_native_compaction_transition_receipt", None)
    if (
        isinstance(latest, NativeCompactionTransitionReceipt)
        and latest.policy.route == route
    ):
        agent._native_compaction_transition_receipt = None


def has_unresolved_native_compaction_failure(
    agent: Any, route: NativeCompactionRoute
) -> bool:
    """Return whether this route has unresolved in-memory custody failure."""
    if bool(getattr(agent, _FAILED_RECEIPTS_OVERFLOW_ATTR, False)):
        return True
    registry = _failed_receipt_registry(agent, create=False)
    if bool(getattr(agent, _FAILED_RECEIPTS_OVERFLOW_ATTR, False)):
        return True
    guard = registry.get(compaction_route_key(route))
    if isinstance(guard, NativeCompactionTransitionReceipt) and guard.failed_closed:
        return True
    latest = getattr(agent, "_native_compaction_transition_receipt", None)
    return bool(
        isinstance(latest, NativeCompactionTransitionReceipt)
        and latest.failed_closed
        and latest.policy.route == route
    )


def effective_auto_compaction_mode(agent: Any) -> str:
    """Return the effective Responses automatic-compaction owner mode."""
    if not bool(getattr(agent, "compression_enabled", True)):
        return "off"
    raw_mode = str(
        getattr(agent, "codex_responses_auto_compaction", "hermes") or "hermes"
    ).strip().lower()
    # Construction validates config. A later malformed mutation must fail
    # closed instead of unexpectedly enabling provider state.
    return raw_mode if raw_mode in {"hermes", "native", "off"} else "hermes"


def configured_codex_responses_auto_compaction_mode(config: Any) -> str:
    """Normalize the configured Responses owner mode without constructing an agent."""
    if not isinstance(config, Mapping):
        return "hermes"
    compression = config.get("compression")
    if not isinstance(compression, Mapping):
        return "hermes"
    raw_mode = str(compression.get("codex_responses_auto", "hermes") or "hermes")
    mode = raw_mode.strip().lower()
    return mode if mode in {"hermes", "native", "off"} else "hermes"


def reconcile_policy_for_current_route(
    agent: Any,
    *,
    provider: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    refresh: bool = False,
) -> NativeCompactionPolicy:
    """Reconcile cached capability state before any ownership decision."""
    route = route_for_request(
        provider=provider if provider is not None else getattr(agent, "provider", ""),
        endpoint=endpoint if endpoint is not None else getattr(agent, "base_url", ""),
        model=model if model is not None else getattr(agent, "model", ""),
    )
    latest_receipt = getattr(agent, "_native_compaction_transition_receipt", None)
    if (
        isinstance(latest_receipt, NativeCompactionTransitionReceipt)
        and latest_receipt.failed_closed
    ):
        record_native_compaction_transition_receipt(agent, latest_receipt)
    cached = getattr(agent, "_native_compaction_policy", None)
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    durable_boundary = session_db is not None and bool(session_id)
    if not durable_boundary:
        active_read_status = getattr(agent, "_native_compaction_read_status", None)
        if (
            isinstance(active_read_status, NativeCompactionReadOutcome)
            and active_read_status.failed_closed
        ):
            # Boundary loss is not a successful reread. Preserve the active
            # guard and same-route custody until a durable read succeeds.
            if isinstance(cached, NativeCompactionPolicy) and cached.route == route:
                return cached
            return NativeCompactionPolicy(route=route)
        agent._native_compaction_read_status = NativeCompactionReadOutcome(
            outcome="not_attempted",
            policy=NativeCompactionPolicy(route=route),
        )
    if (
        not durable_boundary
        and isinstance(cached, NativeCompactionPolicy)
        and cached.route == route
        and cached.capability in OWNING_CAPABILITY_STATES
    ):
        if has_unresolved_native_compaction_failure(agent, route):
            return cached
        receipt = failed_closed_transition_receipt(
            cached,
            error="durable_session_boundary_missing",
        )
        record_native_compaction_transition_receipt(agent, receipt)
        agent._native_compaction_policy = receipt.policy
        return receipt.policy
    durable_refresh = bool(session_db is not None and session_id)
    if (
        not durable_refresh
        and isinstance(cached, NativeCompactionPolicy)
        and cached.route == route
    ):
        return cached
    read_outcome = read_policy_for_route(
        session_db,
        session_id,
        route,
    )
    agent._native_compaction_read_status = read_outcome
    if read_outcome.failed_closed:
        if (
            isinstance(cached, NativeCompactionPolicy)
            and cached.route == route
        ):
            return cached
        policy = NativeCompactionPolicy(route=route)
        agent._native_compaction_policy = policy
        return policy
    assert read_outcome.policy is not None
    policy = read_outcome.policy
    # Only a successful durable read supersedes unresolved custody, and only
    # for the route proven by that read. ``not_attempted`` preserves guards.
    resolve_native_compaction_receipt_for_durable_read(agent, read_outcome)
    agent._native_compaction_policy = policy
    return policy


def native_compaction_read_failed(agent: Any) -> bool:
    """Return the canonical fail-closed durable-read guard for an agent."""
    status = getattr(agent, "_native_compaction_read_status", None)
    return bool(
        isinstance(status, NativeCompactionReadOutcome)
        and status.failed_closed
    )


def should_defer_automatic_hermes_compaction(
    agent: Any, *, refresh: bool = False
) -> bool:
    """Whether the current Responses route forbids automatic Hermes compaction.

    A route can retain native ownership after configuration switches to
    ``hermes``. Automatic textual compression therefore remains deferred;
    only an explicit one-shot manual or emergency proof may hand off custody.
    """
    if getattr(agent, "api_mode", None) != "codex_responses":
        return False
    policy = reconcile_policy_for_current_route(agent, refresh=refresh)
    if native_compaction_read_failed(agent):
        return True
    if has_unresolved_native_compaction_failure(agent, policy.route):
        return True
    mode = effective_auto_compaction_mode(agent)
    return should_defer_hermes_compaction(
        mode=mode,
        policy=policy,
    )


def _handoff_owned_route_to_hermes(
    agent: Any,
    policy: NativeCompactionPolicy,
    *,
    reason: str,
) -> NativeCompactionPolicy | bool:
    """Durably quarantine an owning route before any Hermes textual rewrite."""
    if policy.capability not in OWNING_CAPABILITY_STATES:
        return policy
    if policy.capability == "quarantined":
        return policy
    desired = replace(
        policy.transition("quarantined", error=reason),
        fallback_count=policy.fallback_count + 1,
    )
    receipt = persist_policy_compare_and_set(
        getattr(agent, "_session_db", None),
        getattr(agent, "session_id", None),
        desired,
    )
    record_native_compaction_transition_receipt(agent, receipt)
    agent._native_compaction_policy = receipt.policy
    if not receipt.authorizes_transition:
        return False
    agent._native_compaction_request_active = False
    agent._native_compaction_replay_attempted = False
    return receipt.policy


def prepare_manual_hermes_compaction(
    agent: Any,
    *,
    reason: str = "explicit_manual_request",
) -> HermesCompactionAuthorization | bool:
    """Prepare one explicit manual rewrite and return route-bound proof.

    The proof is registered by object identity on the agent and may be consumed
    exactly once by ``AIAgent._compress_context``. ``off`` is an absolute
    textual-compression ban and cannot mint a manual proof.
    """
    mode = effective_auto_compaction_mode(agent)
    if mode == "off":
        return False
    setattr(agent, _MANUAL_HERMES_AUTHORIZATION_ATTR, None)
    route = route_for_request(
        provider=str(getattr(agent, "provider", "") or ""),
        endpoint=str(getattr(agent, "base_url", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    if getattr(agent, "api_mode", None) == "codex_responses":
        policy = reconcile_policy_for_current_route(agent, refresh=True)
        status = getattr(agent, "_native_compaction_read_status", None)
        if native_compaction_read_failed(agent):
            return False
        if has_unresolved_native_compaction_failure(agent, policy.route):
            return False
        if mode in {"native", "off"} and not (
            isinstance(status, NativeCompactionReadOutcome) and status.succeeded
        ):
            return False
        handed_off = _handoff_owned_route_to_hermes(
            agent,
            policy,
            reason=reason,
        )
        if handed_off is False:
            return False
        assert isinstance(handed_off, NativeCompactionPolicy)
        policy = handed_off
        route = policy.route
        revision = policy.revision
    else:
        revision = 0
    authorization = HermesCompactionAuthorization(
        route=route,
        session_id=getattr(agent, "session_id", None),
        policy_revision=revision,
        mode=mode,
        reason=reason,
    )
    setattr(agent, _MANUAL_HERMES_AUTHORIZATION_ATTR, authorization)
    return authorization


def consume_manual_hermes_compaction_authorization(
    agent: Any,
    authorization: Any,
) -> bool:
    """Consume exactly one matching manual authorization, failing closed."""
    registered = getattr(agent, _MANUAL_HERMES_AUTHORIZATION_ATTR, None)
    if registered is not authorization or not isinstance(
        authorization, HermesCompactionAuthorization
    ):
        return False
    setattr(agent, _MANUAL_HERMES_AUTHORIZATION_ATTR, None)
    route = route_for_request(
        provider=str(getattr(agent, "provider", "") or ""),
        endpoint=str(getattr(agent, "base_url", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    if authorization.route != route:
        return False
    if authorization.session_id != getattr(agent, "session_id", None):
        return False
    if authorization.mode == "off":
        return False
    if authorization.mode != effective_auto_compaction_mode(agent):
        return False
    if getattr(agent, "api_mode", None) != "codex_responses":
        return True
    policy = getattr(agent, "_native_compaction_policy", None)
    if not isinstance(policy, NativeCompactionPolicy) or policy.route != route:
        return False
    if policy.revision != authorization.policy_revision:
        return False
    if native_compaction_read_failed(agent):
        return False
    if has_unresolved_native_compaction_failure(agent, route):
        return False
    if authorization.mode in {"native", "off"}:
        status = getattr(agent, "_native_compaction_read_status", None)
        if not isinstance(status, NativeCompactionReadOutcome) or not status.succeeded:
            return False
    return True


def prepare_emergency_hermes_compaction(
    agent: Any,
    *,
    reason: str,
) -> EmergencyHermesCompactionAuthorization | bool:
    """Prepare one overflow rewrite after resolving durable route custody."""
    mode = effective_auto_compaction_mode(agent)
    if mode == "off":
        return False
    setattr(agent, _EMERGENCY_HERMES_AUTHORIZATION_ATTR, None)
    route = route_for_request(
        provider=str(getattr(agent, "provider", "") or ""),
        endpoint=str(getattr(agent, "base_url", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    if getattr(agent, "api_mode", None) == "codex_responses":
        policy = reconcile_policy_for_current_route(agent, refresh=True)
        status = getattr(agent, "_native_compaction_read_status", None)
        if native_compaction_read_failed(agent):
            return False
        if has_unresolved_native_compaction_failure(agent, policy.route):
            return False
        if mode == "native" and not (
            isinstance(status, NativeCompactionReadOutcome) and status.succeeded
        ):
            return False
        handed_off = _handoff_owned_route_to_hermes(
            agent,
            policy,
            reason=reason,
        )
        if not isinstance(handed_off, NativeCompactionPolicy):
            return False
        policy = handed_off
        route = policy.route
        revision = policy.revision
    else:
        revision = 0
    authorization = EmergencyHermesCompactionAuthorization(
        route=route,
        session_id=getattr(agent, "session_id", None),
        policy_revision=revision,
        mode=mode,
        reason=reason,
    )
    setattr(agent, _EMERGENCY_HERMES_AUTHORIZATION_ATTR, authorization)
    return authorization


def consume_emergency_hermes_compaction_authorization(
    agent: Any,
    authorization: Any,
) -> bool:
    """Consume exactly one matching overflow authorization, failing closed."""
    registered = getattr(agent, _EMERGENCY_HERMES_AUTHORIZATION_ATTR, None)
    if registered is not authorization or not isinstance(
        authorization, EmergencyHermesCompactionAuthorization
    ):
        return False
    setattr(agent, _EMERGENCY_HERMES_AUTHORIZATION_ATTR, None)
    route = route_for_request(
        provider=str(getattr(agent, "provider", "") or ""),
        endpoint=str(getattr(agent, "base_url", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    if authorization.route != route:
        return False
    if authorization.session_id != getattr(agent, "session_id", None):
        return False
    if authorization.mode == "off":
        return False
    if authorization.mode != effective_auto_compaction_mode(agent):
        return False
    if getattr(agent, "api_mode", None) != "codex_responses":
        return True
    policy = getattr(agent, "_native_compaction_policy", None)
    if not isinstance(policy, NativeCompactionPolicy) or policy.route != route:
        return False
    if policy.revision != authorization.policy_revision:
        return False
    if native_compaction_read_failed(agent):
        return False
    if has_unresolved_native_compaction_failure(agent, route):
        return False
    if authorization.mode == "native":
        status = getattr(agent, "_native_compaction_read_status", None)
        if not isinstance(status, NativeCompactionReadOutcome) or not status.succeeded:
            return False
    return True


def has_replayable_compaction_sidecar(
    messages: Any,
    *,
    route: NativeCompactionRoute,
    expected_digest: Optional[str] = None,
) -> bool:
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        try:
            validated = validate_compaction_message_sidecar(message)
        except NativeCompactionStateError:
            continue
        if validated is None:
            continue
        _, sidecar_route, digest = validated
        if sidecar_route != route or digest is None:
            continue
        if expected_digest is None or digest == expected_digest:
            return True
    return False


def stage_native_compaction_checkpoint(
    agent: Any, message: Mapping[str, Any]
) -> None:
    """Bind a pending policy transition to the exact checkpoint message.

    Durable sessions defer ``item_observed`` until SQLite commits the ordered
    output sidecar and the v3 ledger in one transaction. A missing durable
    boundary fails closed; it never grants acknowledgement-free ownership.
    """
    policy = getattr(agent, "_native_compaction_pending_policy", None)
    if not isinstance(policy, NativeCompactionPolicy):
        return
    if not has_replayable_compaction_sidecar(
        [message],
        route=policy.route,
        expected_digest=policy.last_compaction_digest,
    ):
        # The response normalizer stamps every ordered item. Refuse to bind a
        # malformed/mismatched checkpoint even for ephemeral sessions; request
        # construction will then keep canonical history and Hermes fallback.
        agent._native_compaction_pending_policy = None
        return
    if getattr(agent, "_session_db", None) is None or not getattr(
        agent, "session_id", None
    ):
        receipt = failed_closed_transition_receipt(
            policy,
            error="durable_session_boundary_missing",
        )
        record_native_compaction_transition_receipt(agent, receipt)
        agent._native_compaction_policy = receipt.policy
        agent._native_compaction_pending_policy = None
        return
    pending = getattr(agent, "_native_compaction_pending_commits", None)
    if not isinstance(pending, dict):
        pending = {}
        agent._native_compaction_pending_commits = pending
    pending[id(message)] = policy


def pending_checkpoint_policy(
    agent: Any, message: Mapping[str, Any]
) -> Optional[NativeCompactionPolicy]:
    pending = getattr(agent, "_native_compaction_pending_commits", None)
    if not isinstance(pending, dict):
        return None
    policy = pending.get(id(message))
    return policy if isinstance(policy, NativeCompactionPolicy) else None


def complete_native_compaction_checkpoint(
    agent: Any,
    message: Mapping[str, Any],
    policy: NativeCompactionPolicy,
    *,
    conflict_reconciled: bool = False,
) -> NativeCompactionTransitionReceipt:
    """Publish a receipt only after the atomic checkpoint+ledger write exists."""
    try:
        ledger = load_compaction_ledger(
            getattr(agent, "_session_db", None),
            getattr(agent, "session_id", None),
        )
        committed = ledger.policy_for(policy.route)
        if (
            committed.capability != policy.capability
            or committed.last_compaction_digest != policy.last_compaction_digest
        ):
            raise NativeCompactionStateError(
                "durable checkpoint transition does not match the staged policy"
            )
        receipt = NativeCompactionTransitionReceipt(
            outcome=(
                "conflict_reconciled" if conflict_reconciled else "committed"
            ),
            policy=committed,
            ledger=ledger,
            durable_revision=ledger.revision,
            attempts=2 if conflict_reconciled else 1,
        )
    except Exception as exc:
        logger.error(
            "Failed to acknowledge native Responses checkpoint durability: %s",
            exc,
        )
        receipt = failed_closed_transition_receipt(
            policy,
            error="checkpoint_durability_unacknowledged",
            attempts=2 if conflict_reconciled else 1,
        )
    record_native_compaction_transition_receipt(agent, receipt)
    agent._native_compaction_policy = receipt.policy
    pending = getattr(agent, "_native_compaction_pending_commits", None)
    if isinstance(pending, dict):
        pending.pop(id(message), None)
    staged = getattr(agent, "_native_compaction_pending_policy", None)
    if staged is policy or staged == policy:
        agent._native_compaction_pending_policy = None
    return receipt
