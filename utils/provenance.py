import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATHS = (Path(__file__).resolve().parent.parent / "configs" / "reactor_params.yaml",)


def _git(*args):
    try:
        result = subprocess.run(["git", *args], cwd=_REPO_DIR, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_commit():
    return _git("rev-parse", "HEAD")


def _git_dirty():
    status = _git("status", "--porcelain")
    return status is None or len(status) > 0


def _relativize_script(script):
    if not Path(script).is_absolute():
        return script
    repo_root = _git("rev-parse", "--show-toplevel")
    if repo_root is None:
        return script
    try:
        return str(Path(script).resolve().relative_to(Path(repo_root).resolve()))
    except ValueError:
        return script


def _config_hash(config_paths):
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in config_paths):
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def save_with_provenance(path, data, params, *, script, start_time=None,
                          config_paths=None, extra=None):
    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]

    if data is not None:
        if len(paths) != 1:
            raise ValueError("data is only supported with a single output path")
        np.savez(paths[0], **data)

    record = {
        "script": _relativize_script(script),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": params,
        "config_hash": _config_hash(config_paths or _DEFAULT_CONFIG_PATHS),
        "outputs": [str(p) for p in paths],
        "duration_s": (time.time() - start_time) if start_time is not None else None,
    }
    if extra:
        record.update(extra)

    sidecar = paths[0].parent / f"{paths[0].stem}.provenance.json"
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    return sidecar
