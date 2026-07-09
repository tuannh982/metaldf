"""CLI entry point: python -m metaldf script.py.

Installs import interception, then runs the user's script.
"""

from __future__ import annotations

import argparse
import runpy
import sys

from metaldf._accelerator import install


def main() -> None:
    """Run the metaldf CLI."""
    parser = argparse.ArgumentParser(
        prog="metaldf",
        description="Run a Python script with metaldf GPU acceleration.",
    )
    parser.add_argument("script", help="Python script to run")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments to pass to the script"
    )
    args = parser.parse_args()

    # Install import interception before running the script
    install()

    # Pass remaining args to the script
    sys.argv = [args.script] + args.args

    # Run the script
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
