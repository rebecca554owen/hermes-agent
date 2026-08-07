"""Auxiliary transport routing for built-in OpenCode providers."""

from unittest.mock import MagicMock, patch

import pytest
import yaml


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_compression_config(tmp_path, model: str, api_mode: str) -> None:
    config = {
        "model": {"default": model, "provider": "opencode-go"},
        "auxiliary": {
            "compression": {
                "provider": "opencode-go",
                "model": model,
                "api_mode": api_mode,
            }
        },
    }
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump(config))


@pytest.mark.parametrize(
    ("model", "stale_mode", "base_url", "expect_responses", "expected_client_url"),
    [
        (
            "deepseek-v4-flash",
            "chat_completions",
            "https://opencode.ai/zen/go",
            True,
            "https://opencode.ai/zen/go/v1",
        ),
        (
            "deepseek-v4-pro",
            "codex_responses",
            "https://opencode.ai/zen/go/v1",
            False,
            "https://opencode.ai/zen/go/v1",
        ),
    ],
)
def test_compression_uses_authoritative_opencode_model_transport(
    tmp_path, model, stale_mode, base_url, expect_responses, expected_client_url
):
    """Task-level api_mode cannot override OpenCode's per-model wire shape."""
    _write_compression_config(tmp_path, model, stale_mode)
    raw_client = MagicMock()

    with (
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "api_key": "go-key",
                "base_url": base_url,
            },
        ),
        patch("agent.auxiliary_client.OpenAI", return_value=raw_client) as openai,
    ):
        from agent.auxiliary_client import (
            CodexAuxiliaryClient,
            get_text_auxiliary_client,
        )

        client, resolved_model = get_text_auxiliary_client(task="compression")

    assert resolved_model == model
    assert openai.call_args.kwargs["base_url"] == expected_client_url
    assert isinstance(client, CodexAuxiliaryClient) is expect_responses
    if not expect_responses:
        assert client is raw_client
