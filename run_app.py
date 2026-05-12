"""Launch the FastAPI backend for AI Visualizer Studio.

Usage (from repo root, with the project's venv active):

    python run_app.py                # listens on 127.0.0.1:8000
    python run_app.py --host 0.0.0.0 --port 9000
    python run_app.py --reload       # dev mode, auto-restart on code changes

Then, in another terminal:

    cd frontend
    npm install        # first time only
    npm run dev        # opens http://localhost:5173
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing the app so os.environ is complete for startup/STT.
_REPO_ROOT = Path(__file__).resolve().parent
# Local: .env overrides pre-set shell vars so comma-separated OPENAI_API_KEY loads fully.
# Render: no file → unchanged; vars come from the dashboard only.
load_dotenv(_REPO_ROOT / ".env", override=True)

from backend.app import app
import uvicorn


def main() -> None:
    default_port = int(os.environ.get("PORT", "8000"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=default_port)
    p.add_argument("--reload", action="store_true", help="Auto-restart on code changes (dev).")
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config for backend prompts pipeline.",
    )
    args = p.parse_args()

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            p.error(f"Config file not found: {config_path}")
        os.environ["APP_CONFIG_PATH"] = str(config_path)

    state_path = Path(__file__).resolve().parent / ".app_state.json"
    if state_path.exists():
        try:
            state_path.unlink()
        except OSError:
            # Non-fatal: app can still run if state cleanup fails.
            pass

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
