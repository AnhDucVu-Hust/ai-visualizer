"""Serve large artifacts via nginx X-Accel-Redirect when enabled.

When ``NGINX_ACCEL_ENABLED`` is set, ``download_job_artifact`` returns a response
with ``X-Accel-Redirect`` so nginx serves the file from disk (sendfile) instead
of streaming through uvicorn. Files must live under the app ``temp/`` directory.

See ``deploy/nginx/README.md`` for the matching nginx ``location`` block.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.responses import FileResponse
from starlette.responses import Response


def nginx_accel_enabled() -> bool:
    return os.getenv("NGINX_ACCEL_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def nginx_accel_internal_prefix() -> str:
    """URI prefix nginx maps to ``temp/`` (must end with ``/``)."""
    raw = os.getenv("NGINX_ACCEL_INTERNAL_PREFIX", "/internal-temp/").strip() or "/internal-temp/"
    if not raw.startswith("/"):
        raw = f"/{raw}"
    if not raw.endswith("/"):
        raw = f"{raw}/"
    return raw


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_file_download_response(
    file_path: Path,
    *,
    download_name: str,
    media_type: str,
    temp_root: Path,
) -> FileResponse | Response:
    """Return nginx accel redirect or a normal ``FileResponse``."""
    resolved = file_path.resolve()
    if nginx_accel_enabled() and _is_under_root(resolved, temp_root):
        rel = resolved.relative_to(temp_root.resolve())
        redirect_uri = f"{nginx_accel_internal_prefix()}{rel.as_posix()}"
        return Response(
            headers={
                "X-Accel-Redirect": redirect_uri,
                "Content-Type": media_type,
                "Content-Disposition": f'attachment; filename="{download_name}"',
            },
        )
    return FileResponse(path=str(resolved), filename=download_name, media_type=media_type)
