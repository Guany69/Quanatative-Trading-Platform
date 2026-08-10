"""YAML configuration loading, merging, hashing, and snapshotting."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from quant_platform.config.models import ResearchCharter

# Matches ${VAR} and ${VAR:-default} so configs can reference secrets without embedding them.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

CONFIG_DIR = Path("configs")


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} references.

    Credentials live in the environment, never in YAML. This lets a config *reference* a
    secret by name while the value stays outside version control.
    """
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            resolved = os.environ.get(var)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"environment variable '{var}' is referenced by config but not set "
                        f"(and no default was given)"
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file with environment expansion."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping in {p}, got {type(raw).__name__}")
    return _expand_env(raw)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_charter(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ResearchCharter:
    """Load and validate the research charter.

    A charter may pull sub-configs from sibling files via an ``includes`` mapping, e.g.::

        includes:
          universe: universe.yaml
          costs: costs.yaml

    Validation errors surface here rather than deep inside a pipeline run.
    """
    if path is None:
        return ResearchCharter(**(overrides or {}))

    p = Path(path)
    raw = load_yaml(p)

    includes = raw.pop("includes", {})
    if includes:
        if not isinstance(includes, dict):
            raise ConfigError(f"'includes' must be a mapping in {p}")
        for section, include_path in includes.items():
            included = load_yaml(p.parent / include_path)
            # An inline section in the charter wins over the included file.
            section_data = deep_merge(included, raw.get(section, {}) or {})
            raw[section] = section_data

    if overrides:
        raw = deep_merge(raw, overrides)

    try:
        return ResearchCharter(**raw)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {p}: {exc}") from exc


def config_hash(charter: ResearchCharter) -> str:
    """Stable short hash of a resolved config.

    Keys are sorted so the hash depends on content, not YAML ordering. Used to tie an
    experiment's results to the exact configuration that produced them.
    """
    payload = json.dumps(charter.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def snapshot_config(charter: ResearchCharter, destination: str | Path) -> Path:
    """Write the fully resolved config beside a run's artifacts.

    Reproducing a run requires the *resolved* config, not the source YAML, because includes,
    env expansion, and defaults all change what actually executed.
    """
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_hash": config_hash(charter), "config": charter.to_dict()}
    with dest.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return dest
