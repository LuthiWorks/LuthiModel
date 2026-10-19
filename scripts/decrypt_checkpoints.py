"""Write a plaintext copy of every encrypted .luthi checkpoint under some folders.

Run it once; after that no checkpoint needs a password:

    python scripts\\decrypt_checkpoints.py E:\\runs E:\\luthi_experiment_archive

For each ``<name>.luthi`` it writes ``<name>.decrypted.pt`` beside it, plus
``<name>.decrypted.json`` (checksums of both files, and which password
worked -- by label, never by value). ``luthi.checkpoint.load_checkpoint``
and ``python -m luthi.generate`` read the copy with no password.

Safe to re-run: originals are only ever read, never modified or deleted
(trained checkpoints are archived, never deleted -- CLAUDE.md), and a file
that already has a copy is skipped.

Passwords tried, in order: each ``--password``; then ``LUTHI_CHECKPOINT_KEY``
if it is set; and, only if neither gave any, one typed at a hidden prompt.
Different runs may have used different keys -- pass every candidate.

JEPA-runner checkpoints (``ckpt_*.pt``) were never encrypted and are not
touched. Exit code 0 when every encrypted file has a copy; 1 when any could
not be decrypted with the passwords given (they are listed).

Why this exists: docs/DECISIONS.md, 2026-10-19 (encryption made optional).
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from luthi.checkpoint import plaintext_copy_path, write_plaintext_copy

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", help="folders to search, recursively")
    ap.add_argument("--password", action="append", default=[],
                    help="a candidate password (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be done; decrypt nothing")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for folder in args.folders:
        root = Path(folder)
        if not root.is_dir():
            print(f"not a folder: {root}", file=sys.stderr)
            return 2
        files.extend(sorted(root.rglob("*.luthi")))
    print(f"{len(files)} .luthi file(s) found")
    if not files:
        return 0
    if args.dry_run:
        for f in files:
            state = "copy exists" if plaintext_copy_path(f).exists() else "would decrypt"
            print(f"  {state:14s} {f}")
        return 0

    candidates = [(f"--password #{i + 1}", p) for i, p in enumerate(args.password)]
    if os.environ.get("LUTHI_CHECKPOINT_KEY"):
        candidates.append(("LUTHI_CHECKPOINT_KEY", os.environ["LUTHI_CHECKPOINT_KEY"]))
    if not candidates:
        candidates.append(("prompt", getpass.getpass("checkpoint password: ")))

    failed: list[Path] = []
    for f in files:
        rec = write_plaintext_copy(f, candidates)
        note = f" ({rec['password_label']})" if "password_label" in rec else ""
        print(f"  {rec['status']:19s} {f}{note}")
        if rec["status"] == "no_password_matched":
            failed.append(f)

    if failed:
        print(
            f"\n{len(failed)} file(s) did not open with the passwords given. "
            "Nothing was written for them and the originals are untouched. "
            "Re-run with another --password.",
            file=sys.stderr,
        )
        return 1
    print("\nDone. Every encrypted checkpoint has a plaintext copy beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
