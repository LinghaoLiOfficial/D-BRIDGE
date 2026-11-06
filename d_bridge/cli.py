from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_settings
from .runtime import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hierarchical incremental semantic label network experiment.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing input JSON post data (defaults to DATA_DIR).")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Environment configuration file (default: .env).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.env_file)
        run_dir = run_experiment(settings, args.data_dir)
        print(f"Results saved to: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
