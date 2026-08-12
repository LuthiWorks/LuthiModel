"""Build `gutenberg_768_filelist.txt` -- the ~113M-token corpus for the 768x8 family.

Brian's data-scaling ruling (2026-07-18, data ~ width^2): 768/512 = 1.5x width
-> 2.25x tokens against the 50.4M the 512 family trained on. Spec:
`docs/research/2026-08-11_768x8-family-spec.md`.

Three things this script refuses to do quietly, each a defect class the record
has paid for:

1. **Dedup keys on the Gutenberg id AND on the content hash.** Two layers,
   because one is provably not enough:
   - *by id*: all 100 files in `corpus_build/gutenberg_100` also exist in
     `E:/data/gutenberg_4gb` under a different path, byte-identical (verified
     100/100). The existing 482-file corpus is clean only because its 382 E:
     picks happened to miss them. A path-keyed selection re-adds those books,
     the effective expansion silently shrinks, and every counter reads healthy.
   - *by sha256*: a full scan of both roots (11,213 files) found 101 duplicate
     groups, and exactly ONE of them carries two different Gutenberg ids --
     PG1133 and PG2269 are the same text catalogued twice. Id-dedup cannot see
     it, and the first build of this list took both. One file is a small loss;
     an unchecked class is not, and the check costs a hash.

2. **The token count is measured, not estimated.** Every selected file is run
   through the project's own BPETokenizer (tokenizer_32k) and counted. A
   bytes-times-ratio estimate is used ONLY to choose candidates; the number
   that goes in the manifest is the measured one.

3. **A missing E: is fatal.** Without it the pool is 100 files and this would
   otherwise build a corpus 10x too small and report success.

The 482 existing files are kept as a SUBSET by construction, so the 512
family's data is nested inside the 768 family's -- one fewer difference
between the two families being compared.

No sanitizer pass is run, deliberately: `download_gutenberg.py` strips
Gutenberg boilerplate at download time and `sanitize_corpus.py` never covered
this corpus (gutenberg is not in its DEFAULT_CORPORA). The pool is therefore
uniformly treated already, and the existing 482 received exactly this
treatment. Running the sanitizer over the new half only would make it cleaner
than the old half -- a confound in the one family whose purpose is comparison
against the old half. Residual transcriber's notes exist at the same rate in
both (measured: 2/80 selected, 10/80 unused) and are left alone.

Usage:
    python corpus_build/build_768_filelist.py --dry-run
    python corpus_build/build_768_filelist.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

POOL = Path("E:/data/gutenberg_4gb")
G100 = REPO / "corpus_build" / "gutenberg_100"
EXISTING = REPO / "corpus_build" / "gutenberg_4x_filelist.txt"
OUT = REPO / "corpus_build" / "gutenberg_768_filelist.txt"
MANIFEST = REPO / "corpus_build" / "gutenberg_768_manifest.json"
TOKENIZER = REPO / "corpus_build" / "tokenizer_32k.json"

TARGET_TOKENS = 113_000_000
SEED = 768                 # recorded: the selection is reproducible
BYTES_PER_TOKEN_GUESS = 0.2676   # measured on the existing 482; candidate sizing only


def pg_id(p: Path | str) -> str:
    """Gutenberg id, case-normalised. The dedup key -- NOT the path."""
    return Path(p).name.split(".")[0].upper()


def _sha256(path_str: str) -> tuple[str, str]:
    import hashlib
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return path_str, h.hexdigest()


def _count_tokens(path_str: str) -> tuple[str, int, int]:
    from luthi.tokenizer import BPETokenizer
    global _TOK
    try:
        _TOK
    except NameError:
        _TOK = BPETokenizer.load(str(TOKENIZER))
    text = open(path_str, encoding="utf-8", errors="replace").read()
    return path_str, len(_TOK.encode(text)), os.path.getsize(path_str)


def to_line(p: Path) -> str:
    """Match the 4x filelist's format: repo-relative for in-repo files,
    absolute for E:, Windows separators."""
    try:
        rel = p.resolve().relative_to(REPO)
        return str(rel).replace("/", "\\")
    except ValueError:
        return str(p).replace("/", "\\")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET_TOKENS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    # -- 3. fail loud on a missing pool ------------------------------------
    if not POOL.is_dir():
        raise SystemExit(
            f"FATAL: {POOL} is not mounted. Refusing to build -- the reachable "
            f"pool would be {G100} alone (100 files, ~10M tokens) and this "
            f"script would report a successful 113M build."
        )

    existing = [l.strip() for l in EXISTING.read_text(encoding="utf-8").splitlines() if l.strip()]
    existing_paths = [Path(str(REPO / e.replace("\\", "/"))) if not e[1:2] == ":"
                      else Path(e.replace("\\", "/")) for e in existing]
    missing = [p for p in existing_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} file(s) in the existing 4x list do not resolve, "
                         f"e.g. {missing[0]}")

    keep_ids = {pg_id(p) for p in existing_paths}
    print(f"existing 4x corpus : {len(existing_paths)} files, {len(keep_ids)} distinct ids")
    if len(keep_ids) != len(existing_paths):
        raise SystemExit("FATAL: the existing list already contains duplicate ids")

    # -- 1. id-keyed candidate pool ----------------------------------------
    pool = sorted(POOL.glob("*.txt"))
    g100_ids = {pg_id(p) for p in G100.glob("*.txt")}
    collisions = sum(1 for p in pool if pg_id(p) in g100_ids)
    candidates = [p for p in pool if pg_id(p) not in keep_ids]
    print(f"E: pool            : {len(pool):,} files")
    print(f"  ids also present in gutenberg_100 : {collisions}  <- excluded by id, not path")
    print(f"  candidates after id-dedup         : {len(candidates):,}")

    # -- choose candidates by ESTIMATED size, then measure -----------------
    have_bytes = sum(p.stat().st_size for p in existing_paths)
    have_est = have_bytes * BYTES_PER_TOKEN_GUESS
    need_est = args.target - have_est
    rng = random.Random(SEED)
    rng.shuffle(candidates)

    # Second dedup key: content. Hash what we keep, then hash each candidate as
    # it is considered and reject any text already present. PG1133/PG2269 are
    # the same book under two ids -- id-dedup passes them both.
    print("\nhashing for content-dedup (the id key cannot see re-catalogued texts)")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        seen_hashes = {h for _, h in ex.map(_sha256, [str(p) for p in existing_paths],
                                            chunksize=16)}
        cand_hashes = dict(ex.map(_sha256, [str(p) for p in candidates], chunksize=32))

    picked, acc, rejected = [], 0.0, 0
    for p in candidates:
        if acc >= need_est * 1.06:      # 6% over, trimmed after measuring
            break
        h = cand_hashes[str(p)]
        if h in seen_hashes:
            rejected += 1
            continue
        seen_hashes.add(h)
        picked.append(p)
        acc += p.stat().st_size * BYTES_PER_TOKEN_GUESS
    print(f"  content-duplicate candidates rejected: {rejected}")
    print(f"\nexisting ~{have_est/1e6:.1f}M est tokens; picking {len(picked)} more "
          f"(~{acc/1e6:.1f}M est) toward {args.target/1e6:.0f}M")

    # -- 2. MEASURE, never estimate, the number that ships ------------------
    todo = [str(p) for p in existing_paths + picked]
    print(f"\ntokenizing {len(todo)} files with {args.workers} workers "
          f"(tokenizer_32k) -- this is the real count, not an estimate")
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path_str, n_tok, n_bytes in ex.map(_count_tokens, todo, chunksize=4):
            counts[path_str] = n_tok
            sizes[path_str] = n_bytes
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} files, {sum(counts.values())/1e6:.1f}M tokens")

    base_total = sum(counts[str(p)] for p in existing_paths)
    print(f"\nmeasured: existing 482 = {base_total/1e6:.2f}M tokens")

    # trim the picked tail to land on target by MEASURED tokens
    total = base_total
    final_extra = []
    for p in picked:
        if total >= args.target:
            break
        final_extra.append(p)
        total += counts[str(p)]
    dropped = len(picked) - len(final_extra)

    final = existing_paths + final_extra
    final_ids = {pg_id(p) for p in final}
    if len(final_ids) != len(final):
        raise SystemExit("FATAL: duplicate ids in the final selection")

    total_bytes = sum(sizes[str(p)] for p in final)
    print(f"\n=== RESULT ===")
    print(f"  files          : {len(final)}  ({len(existing_paths)} kept + {len(final_extra)} new)")
    print(f"  MEASURED tokens: {total:,}  ({total/1e6:.2f}M)   target {args.target/1e6:.0f}M")
    print(f"  bytes          : {total_bytes/1e6:.1f} MB")
    print(f"  tokens/file    : {total/len(final)/1e3:.1f}K")
    print(f"  expansion      : {total/base_total:.3f}x the 512 family's corpus "
          f"(ruling asks 2.25x)")
    print(f"  distinct ids   : {len(final_ids)}  (duplicates: 0)")
    print(f"  candidates measured but dropped after trim: {dropped}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    OUT.write_text("\r\n".join(to_line(p) for p in final) + "\r\n", encoding="utf-8", newline="")
    MANIFEST.write_text(json.dumps({
        "built": "2026-08-11",
        "purpose": "768x8 family (stage 55) -- data ~ width^2 ruling, 2.25x of 50.4M",
        "target_tokens": args.target,
        "measured_tokens": total,
        "measured_tokens_existing_482": base_total,
        "expansion_ratio": round(total / base_total, 4),
        "files": len(final),
        "files_kept_from_4x": len(existing_paths),
        "files_added": len(final_extra),
        "bytes": total_bytes,
        "tokenizer": "corpus_build/tokenizer_32k.json",
        "selection_seed": SEED,
        "dedup_keys": [
            "gutenberg id (PGxxxxx), case-normalised, across both roots",
            "sha256 of file content (catches the same text re-catalogued under "
            "a second id, e.g. PG1133 == PG2269)",
        ],
        "gutenberg_100_ids_also_in_pool": collisions,
        "content_duplicate_candidates_rejected": rejected,
        "sanitizer_pass": "none -- pool is boilerplate-stripped at download time; "
                          "see module docstring",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
