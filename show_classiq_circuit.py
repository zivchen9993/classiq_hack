"""Open a saved Classiq temporal-antenna program in Classiq's visualizer.

Requires the Classiq SDK.  Use this in the authenticated hackathon environment
to produce the *actual* Classiq circuit screenshot for the deck.  The repository
also contains static resource and logical-structure figures that do not pretend
to be screenshots of the Classiq UI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?", default="results/classiq_tiny_validation_2026_09_09/three_step/classiq_program.json")
    args = ap.parse_args()
    try:
        from classiq import show
    except ImportError as exc:
        raise SystemExit("Classiq SDK is not installed. Install/authenticate Classiq, then rerun this script.") from exc

    # Modern Classiq show() accepts the serialized synthesized program string.
    # Keeping the exact JSON text preserves the program_id and synthesized data.
    text = Path(args.program).read_text(encoding="utf-8")
    data = json.loads(text)
    program_id = data.get("program_id")
    print(f"Opening saved synthesized Classiq program {program_id or '(unknown id)'}")
    show(text)


if __name__ == "__main__":
    main()
