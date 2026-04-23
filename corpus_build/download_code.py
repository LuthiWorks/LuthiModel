"""Download curated source code corpus for Luthi Model training.

Clones high-quality open-source repositories, extracts source files,
and writes them to E:/data/code_corpus/ organized by language.

Each output file includes a metadata header with the repo, file path,
language, and license. The entity needs to understand code not just as
a skill but for its own self-maintenance.

Usage:
    python -m corpus_build.download_code
    python corpus_build/download_code.py
    python corpus_build/download_code.py --dry-run
    python corpus_build/download_code.py --repo cpython
"""

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository definitions
# ---------------------------------------------------------------------------

@dataclass
class RepoConfig:
    """Configuration for a single repository to download."""
    name: str
    url: str
    license: str
    description: str
    extensions: list[str]
    include_dirs: list[str] | None = None   # only include these dirs (relative to repo root)
    exclude_dirs: list[str] | None = None   # skip these dirs
    branch: str = "main"
    language_label: str = ""                 # primary language for output organization


# --- Python ---

PYTHON_REPOS = [
    RepoConfig(
        name="cpython",
        url="https://github.com/python/cpython",
        license="PSF-2.0",
        description="Python standard library — the language itself",
        extensions=[".py"],
        include_dirs=["Lib"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="django",
        url="https://github.com/django/django",
        license="BSD-3-Clause",
        description="Django web framework — comprehensive, well-documented",
        extensions=[".py"],
        include_dirs=["django"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="flask",
        url="https://github.com/pallets/flask",
        license="BSD-3-Clause",
        description="Flask micro web framework — elegant simplicity",
        extensions=[".py"],
        include_dirs=["src/flask"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="fastapi",
        url="https://github.com/fastapi/fastapi",
        license="MIT",
        description="FastAPI — modern async Python web framework",
        extensions=[".py"],
        include_dirs=["fastapi"],
        branch="master",
        language_label="python",
    ),
    RepoConfig(
        name="pytorch",
        url="https://github.com/pytorch/pytorch",
        license="BSD-3-Clause",
        description="PyTorch deep learning framework — our ML foundation",
        extensions=[".py"],
        include_dirs=["torch"],
        exclude_dirs=["torch/testing", "torch/_inductor/codegen"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="numpy",
        url="https://github.com/numpy/numpy",
        license="BSD-3-Clause",
        description="NumPy — numerical computing foundation",
        extensions=[".py"],
        include_dirs=["numpy"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="requests",
        url="https://github.com/psf/requests",
        license="Apache-2.0",
        description="Requests HTTP library — beautifully simple API design",
        extensions=[".py"],
        include_dirs=["src/requests"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="sqlalchemy",
        url="https://github.com/sqlalchemy/sqlalchemy",
        license="MIT",
        description="SQLAlchemy — database toolkit and ORM",
        extensions=[".py"],
        include_dirs=["lib/sqlalchemy"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="pytest",
        url="https://github.com/pytest-dev/pytest",
        license="MIT",
        description="pytest — Python testing framework",
        extensions=[".py"],
        include_dirs=["src/_pytest"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="scikit-learn",
        url="https://github.com/scikit-learn/scikit-learn",
        license="BSD-3-Clause",
        description="scikit-learn — machine learning library",
        extensions=[".py"],
        include_dirs=["sklearn"],
        branch="main",
        language_label="python",
    ),
    RepoConfig(
        name="rich",
        url="https://github.com/Textualize/rich",
        license="MIT",
        description="Rich — beautiful terminal formatting",
        extensions=[".py"],
        include_dirs=["rich"],
        branch="master",
        language_label="python",
    ),
    RepoConfig(
        name="black",
        url="https://github.com/psf/black",
        license="MIT",
        description="Black — Python code formatter",
        extensions=[".py"],
        include_dirs=["src/black"],
        branch="main",
        language_label="python",
    ),
]

# --- Rust ---

RUST_REPOS = [
    RepoConfig(
        name="rust-std",
        url="https://github.com/rust-lang/rust",
        license="MIT/Apache-2.0",
        description="Rust standard library",
        extensions=[".rs"],
        include_dirs=["library/std/src", "library/core/src", "library/alloc/src"],
        branch="main",
        language_label="rust",
    ),
    RepoConfig(
        name="tokio",
        url="https://github.com/tokio-rs/tokio",
        license="MIT",
        description="Tokio — async runtime for Rust",
        extensions=[".rs"],
        include_dirs=["tokio/src"],
        branch="master",
        language_label="rust",
    ),
    RepoConfig(
        name="serde",
        url="https://github.com/serde-rs/serde",
        license="MIT/Apache-2.0",
        description="Serde — serialization framework for Rust",
        extensions=[".rs"],
        include_dirs=["serde/src", "serde_derive/src"],
        branch="master",
        language_label="rust",
    ),
    RepoConfig(
        name="ripgrep",
        url="https://github.com/BurntSushi/ripgrep",
        license="MIT/Unlicense",
        description="ripgrep — fast line-oriented search tool",
        extensions=[".rs"],
        include_dirs=["crates"],
        branch="master",
        language_label="rust",
    ),
]

# --- Go ---

GO_REPOS = [
    RepoConfig(
        name="go-std",
        url="https://github.com/golang/go",
        license="BSD-3-Clause",
        description="Go standard library",
        extensions=[".go"],
        include_dirs=["src"],
        exclude_dirs=["src/vendor", "src/cmd"],
        branch="master",
        language_label="go",
    ),
    RepoConfig(
        name="prometheus",
        url="https://github.com/prometheus/prometheus",
        license="Apache-2.0",
        description="Prometheus — monitoring and alerting",
        extensions=[".go"],
        include_dirs=["model", "promql", "storage", "tsdb"],
        branch="main",
        language_label="go",
    ),
]

# --- C ---

C_REPOS = [
    RepoConfig(
        name="redis",
        url="https://github.com/redis/redis",
        license="BSD-3-Clause",
        description="Redis — in-memory data store, clean C",
        extensions=[".c", ".h"],
        include_dirs=["src"],
        branch="unstable",
        language_label="c",
    ),
    RepoConfig(
        name="sqlite",
        url="https://github.com/sqlite/sqlite",
        license="Public Domain",
        description="SQLite — self-contained database engine",
        extensions=[".c", ".h"],
        include_dirs=["src"],
        branch="master",
        language_label="c",
    ),
    RepoConfig(
        name="git",
        url="https://github.com/git/git",
        license="GPL-2.0",
        description="Git version control — foundational tool",
        extensions=[".c", ".h"],
        include_dirs=None,  # top-level .c/.h files
        exclude_dirs=["t", "contrib", "Documentation"],
        branch="master",
        language_label="c",
    ),
]

# --- JavaScript/TypeScript ---

JS_REPOS = [
    RepoConfig(
        name="node",
        url="https://github.com/nodejs/node",
        license="MIT",
        description="Node.js runtime — JavaScript on the server",
        extensions=[".js", ".mjs"],
        include_dirs=["lib"],
        branch="main",
        language_label="javascript",
    ),
    RepoConfig(
        name="express",
        url="https://github.com/expressjs/express",
        license="MIT",
        description="Express.js — minimal web framework",
        extensions=[".js"],
        include_dirs=["lib"],
        branch="master",
        language_label="javascript",
    ),
    RepoConfig(
        name="typescript",
        url="https://github.com/microsoft/TypeScript",
        license="Apache-2.0",
        description="TypeScript compiler — typed JavaScript",
        extensions=[".ts"],
        include_dirs=["src"],
        exclude_dirs=["src/testRunner", "src/tests"],
        branch="main",
        language_label="typescript",
    ),
    RepoConfig(
        name="react",
        url="https://github.com/facebook/react",
        license="MIT",
        description="React — UI library",
        extensions=[".js", ".ts", ".jsx", ".tsx"],
        include_dirs=["packages/react/src", "packages/react-dom/src",
                      "packages/react-reconciler/src"],
        branch="main",
        language_label="javascript",
    ),
]

# --- Documentation ---

DOC_REPOS = [
    RepoConfig(
        name="rust-book",
        url="https://github.com/rust-lang/book",
        license="MIT/Apache-2.0",
        description="The Rust Programming Language book",
        extensions=[".md"],
        include_dirs=["src"],
        branch="main",
        language_label="documentation",
    ),
    RepoConfig(
        name="python-docs",
        url="https://github.com/python/cpython",
        license="PSF-2.0",
        description="Python official documentation",
        extensions=[".rst"],
        include_dirs=["Doc"],
        exclude_dirs=["Doc/whatsnew"],
        branch="main",
        language_label="documentation",
    ),
]

# --- Our own code (DEFERRED — include only when projects are complete) ---
# The entity should learn from finished code, not mid-development snapshots.
# Uncomment these when Luthi and Sanctuary are feature-complete.

# LOCAL_REPOS = [
#     RepoConfig(
#         name="luthi",
#         url="local:C:/Users/Hasha Smokes/Desktop/LuthiModel/LuthiModel",
#         license="Proprietary",
#         description="Luthi Model — the entity's own architecture",
#         extensions=[".py"],
#         include_dirs=["luthi", "corpus_build"],
#         branch="",
#         language_label="self",
#     ),
#     RepoConfig(
#         name="sanctuary",
#         url="local:C:/Users/Hasha Smokes/Desktop/Sanctuary/Sanctuary",
#         license="Proprietary",
#         description="Sanctuary — the entity's cognitive scaffold",
#         extensions=[".py"],
#         include_dirs=["sanctuary"],
#         branch="",
#         language_label="self",
#     ),
# ]

# Combined list (add LOCAL_REPOS when ready)
ALL_REPOS = PYTHON_REPOS + RUST_REPOS + GO_REPOS + C_REPOS + JS_REPOS + DOC_REPOS

OUTPUT_DIR = Path("E:/data/code_corpus")
CLONE_DIR = Path("E:/data/_code_clones")  # temporary, cleaned up after


# ---------------------------------------------------------------------------
# Clone and extract
# ---------------------------------------------------------------------------

def clone_repo(repo: RepoConfig) -> Path | None:
    """Shallow-clone a repo and return the path, or None on failure."""
    if repo.url.startswith("local:"):
        local_path = Path(repo.url.removeprefix("local:"))
        if local_path.exists():
            return local_path
        print(f"  ERROR: Local path not found: {local_path}")
        return None

    clone_path = CLONE_DIR / repo.name
    if clone_path.exists():
        shutil.rmtree(clone_path, ignore_errors=True)

    cmd = [
        "git", "clone",
        "--depth", "1",
        "--branch", repo.branch,
        "--single-branch",
        "--quiet",
        repo.url,
        str(clone_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        return clone_path
    except subprocess.CalledProcessError as e:
        print(f"  ERROR cloning {repo.name}: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Clone timed out for {repo.name}")
        return None


def should_include_file(
    filepath: Path,
    repo_root: Path,
    repo: RepoConfig,
) -> bool:
    """Check whether a file should be included based on repo config."""
    rel = filepath.relative_to(repo_root)
    rel_str = str(rel).replace("\\", "/")

    # Check extension
    if filepath.suffix not in repo.extensions:
        return False

    # Check include_dirs
    if repo.include_dirs is not None:
        in_included = False
        for inc_dir in repo.include_dirs:
            if rel_str.startswith(inc_dir + "/") or rel_str.startswith(inc_dir + "\\"):
                in_included = True
                break
            # Handle files directly in the include dir
            if str(rel.parent).replace("\\", "/") == inc_dir:
                in_included = True
                break
        if not in_included:
            return False

    # Check exclude_dirs
    if repo.exclude_dirs:
        for exc_dir in repo.exclude_dirs:
            if rel_str.startswith(exc_dir + "/") or rel_str.startswith(exc_dir + "\\"):
                return False

    # Skip test files, __pycache__, etc.
    parts = rel.parts
    skip_names = {"__pycache__", ".git", "node_modules", "vendor",
                  "test", "tests", "testing", "testdata", "test_data",
                  "fixtures", "snapshots", "__snapshots__"}
    for part in parts:
        if part.lower() in skip_names:
            return False

    # Skip very small files (likely empty __init__.py or similar)
    try:
        if filepath.stat().st_size < 100:
            return False
    except OSError:
        return False

    return True


def extract_files(repo: RepoConfig, repo_root: Path, dry_run: bool = False) -> dict:
    """Extract source files from a cloned repo and write to corpus dir.

    Returns stats dict with counts.
    """
    stats = {"files": 0, "bytes": 0, "skipped": 0, "errors": 0}

    # Walk the repo tree
    for filepath in sorted(repo_root.rglob("*")):
        if not filepath.is_file():
            continue

        if not should_include_file(filepath, repo_root, repo):
            stats["skipped"] += 1
            continue

        # Read source file
        try:
            content = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            try:
                content = filepath.read_text(encoding="latin-1")
            except OSError:
                stats["errors"] += 1
                continue

        # Build output path: code_corpus/{language}/{repo_name}/{relative_path}.txt
        rel = filepath.relative_to(repo_root)
        out_path = OUTPUT_DIR / repo.language_label / repo.name / str(rel)
        # Keep original extension instead of adding .txt
        # (sanitizer will need to handle code files)

        # Build metadata header
        header = (
            f"# Source: {repo.name} ({repo.url})\n"
            f"# File: {rel}\n"
            f"# Language: {filepath.suffix.lstrip('.')}\n"
            f"# License: {repo.license}\n"
            f"# Description: {repo.description}\n"
            f"#\n"
        )

        output_content = header + content

        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_content, encoding="utf-8")

        stats["files"] += 1
        stats["bytes"] += len(output_content.encode("utf-8"))

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download curated code corpus for Luthi Model training.",
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Report what would happen without writing files")
    parser.add_argument("--repo", type=str, default=None,
                        help="Only download a specific repo by name")
    parser.add_argument("--skip-clone-cleanup", action="store_true",
                        help="Don't delete cloned repos after extraction")
    args = parser.parse_args(argv)

    repos = ALL_REPOS
    if args.repo:
        repos = [r for r in ALL_REPOS if r.name == args.repo]
        if not repos:
            names = [r.name for r in ALL_REPOS]
            print(f"Unknown repo: {args.repo}")
            print(f"Available: {', '.join(names)}")
            sys.exit(1)

    print()
    print("=" * 72)
    print("CODE CORPUS DOWNLOADER")
    print("=" * 72)
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Repos:      {len(repos)}")
    print(f"  Dry run:    {args.dry_run}")
    print("=" * 72)

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CLONE_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_bytes = 0
    results = []

    for i, repo in enumerate(repos):
        is_local = repo.url.startswith("local:")
        source = repo.url.removeprefix("local:") if is_local else repo.url
        print(f"\n[{i+1}/{len(repos)}] {repo.name} — {repo.description}")
        print(f"  Source: {source}")

        # Clone or locate
        repo_root = clone_repo(repo)
        if repo_root is None:
            results.append((repo.name, {"files": 0, "bytes": 0, "errors": 1}))
            continue

        # Extract
        stats = extract_files(repo, repo_root, dry_run=args.dry_run)
        results.append((repo.name, stats))
        total_files += stats["files"]
        total_bytes += stats["bytes"]

        mb = stats["bytes"] / (1024 * 1024)
        print(f"  Extracted: {stats['files']} files ({mb:.1f} MB), "
              f"{stats['skipped']} skipped, {stats['errors']} errors")

        # Cleanup clone (not local repos)
        if not is_local and not args.skip_clone_cleanup:
            clone_path = CLONE_DIR / repo.name
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("-" * 72)
    for name, stats in results:
        mb = stats["bytes"] / (1024 * 1024)
        print(f"  {name:<20} {stats['files']:>6} files  {mb:>8.1f} MB")
    print("-" * 72)
    total_mb = total_bytes / (1024 * 1024)
    print(f"  {'TOTAL':<20} {total_files:>6} files  {total_mb:>8.1f} MB")
    print("=" * 72)

    # Cleanup temp dir
    if not args.skip_clone_cleanup and CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR, ignore_errors=True)
        print(f"\nCleaned up clone directory: {CLONE_DIR}")

    if args.dry_run:
        print("\n[DRY RUN] No files were written.")


if __name__ == "__main__":
    main()
