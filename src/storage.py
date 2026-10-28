"""Object storage abstraction.

Backends, chosen from the environment:

* ``S3_BUCKET`` set              -> AWS S3 through the AWS CLI.
* ``S3_BUCKET`` + ``AWS_ENDPOINT_URL`` -> any S3-compatible store (e.g. MinIO).
* ``S3_BUCKET`` unset            -> a local folder (``local_s3/`` or ``LOCAL_S3_DIR``)
                                    with the same key layout, so everything runs offline.

Keys are always S3-style (``raw/batch=2026-01/transactions.parquet``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from src.utils import ROOT, get_logger

log = get_logger("storage")


def bucket() -> str | None:
    value = os.environ.get("S3_BUCKET", "").strip()
    return value or None


def endpoint_url() -> str | None:
    value = os.environ.get("AWS_ENDPOINT_URL", "").strip()
    return value or None


def local_root() -> Path:
    return Path(os.environ.get("LOCAL_S3_DIR") or ROOT / "local_s3")


def is_remote() -> bool:
    return bucket() is not None


def uri(key: str = "") -> str:
    """Human-readable location of a key for the active backend."""
    if is_remote():
        return f"s3://{bucket()}/{key}"
    return str(local_root() / key)


def describe() -> str:
    if not is_remote():
        return f"local folder {local_root()}"
    where = f" via {endpoint_url()}" if endpoint_url() else ""
    return f"s3://{bucket()}{where}"


def aws_cmd(*args: str) -> list[str]:
    exe = shutil.which("aws")
    if exe is None:
        raise RuntimeError(
            "S3_BUCKET is set but the AWS CLI was not found on PATH. "
            "Install AWS CLI v2 or unset S3_BUCKET to use the local fallback."
        )
    cmd = [exe, "s3", *args]
    if endpoint_url():
        cmd += ["--endpoint-url", endpoint_url()]
    return cmd


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    log.debug("running %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def sync_down(prefix: str, dest: Path) -> None:
    """Mirror everything under ``prefix`` into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    if is_remote():
        proc = _run(aws_cmd("sync", uri(prefix), str(dest), "--only-show-errors"), check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"aws s3 sync failed: {proc.stderr.strip()}")
        return
    src = local_root() / prefix
    if not src.is_dir():
        return
    shutil.copytree(src, dest, dirs_exist_ok=True)


def upload(src: Path, key: str) -> None:
    if is_remote():
        _run(aws_cmd("cp", str(src), uri(key), "--only-show-errors"))
        return
    target = local_root() / key
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)


def exists(key: str) -> bool:
    if is_remote():
        proc = _run(aws_cmd("ls", uri(key)), check=False)
        # `aws s3 ls` matches by prefix, so confirm the exact name is listed.
        name = key.rstrip("/").rsplit("/", 1)[-1]
        return proc.returncode == 0 and any(
            line.split()[-1].rstrip("/") == name for line in proc.stdout.splitlines() if line.strip()
        )
    return (local_root() / key).exists()


def download(key: str, dest: Path) -> bool:
    """Download one object. Returns False when it does not exist."""
    if not exists(key):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_remote():
        _run(aws_cmd("cp", uri(key), str(dest), "--only-show-errors"))
    else:
        shutil.copy2(local_root() / key, dest)
    return True


def copy_prefix(src_prefix: str, dst_prefix: str) -> None:
    """Server-side copy of every object under one prefix to another."""
    if is_remote():
        _run(aws_cmd("cp", uri(src_prefix), uri(dst_prefix), "--recursive", "--only-show-errors"))
        return
    src = local_root() / src_prefix
    if src.is_dir():
        shutil.copytree(src, local_root() / dst_prefix, dirs_exist_ok=True)


def list_dirs(prefix: str) -> list[str]:
    """Names of the immediate "sub-folders" under ``prefix`` (e.g. ``batch=2026-01``)."""
    if is_remote():
        proc = _run(aws_cmd("ls", uri(prefix)), check=False)
        if proc.returncode != 0:
            return []
        return sorted(
            line.split()[-1].rstrip("/") for line in proc.stdout.splitlines() if line.strip().startswith("PRE ")
        )
    base = local_root() / prefix
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())
