"""Contract tests for Docker support of the Codex app-server provider."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_codex_cli():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "nodejs" in dockerfile
    assert "npm" in dockerfile
    assert "npm install -g @openai/codex" in dockerfile


def test_dockerfile_prepares_codex_home_for_appuser():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "/home/appuser/.codex" in dockerfile
    assert "appuser" in dockerfile


def test_compose_mounts_host_codex_profile_and_container_cwd():
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "${HOME}/.codex:/home/appuser/.codex" in compose
    assert "CODEX_APP_SERVER_COMMAND=codex" in compose
    assert "CODEX_APP_SERVER_CWD=/home/appuser/app" in compose
