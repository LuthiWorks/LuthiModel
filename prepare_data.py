"""Prepare training data from raw Project Gutenberg downloads.

Strips headers/footers and writes clean text files to data/.
"""

from pathlib import Path

data_dir = Path("data")


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Find start: look for *** START OF ... *** line
    lines = text.split("\n")
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if "*** START OF" in line and "GUTENBERG" in line:
            start_idx = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if "*** END OF" in line and "GUTENBERG" in lines[i]:
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def clean_velveteen_rabbit(text: str) -> str:
    """Extract just the story from The Velveteen Rabbit.
    Story starts with 'HERE was once a velveteen rabbit'."""
    # Find story start
    idx = text.find("HERE was once a velveteen rabbit")
    if idx == -1:
        # Try alternate
        idx = text.find("There was once a velveteen rabbit")
    if idx != -1:
        text = text[idx:]
    return text.strip()


# Process each file
files_to_process = {
    "velveteen_rabbit_raw.txt": None,  # No raw, already in clean form
    "house_pooh_corner_raw.txt": "house_pooh_corner.txt",
}

# 1. Fix Velveteen Rabbit (already downloaded but has extra header)
vr_path = data_dir / "velveteen_rabbit.txt"
if vr_path.exists():
    text = vr_path.read_text(encoding="utf-8")
    clean = clean_velveteen_rabbit(text)
    vr_path.write_text(clean, encoding="utf-8")
    print(f"  velveteen_rabbit.txt: {len(clean):,} chars")

# 2. Fix Winnie the Pooh (has Gutenberg header)
wtp_path = data_dir / "winnie_the_pooh.txt"
if wtp_path.exists():
    text = wtp_path.read_text(encoding="utf-8")
    clean = strip_gutenberg(text)
    wtp_path.write_text(clean, encoding="utf-8")
    print(f"  winnie_the_pooh.txt: {len(clean):,} chars")

# 3. Process House at Pooh Corner (raw)
raw_path = data_dir / "house_pooh_corner_raw.txt"
clean_path = data_dir / "house_pooh_corner.txt"
if raw_path.exists():
    text = raw_path.read_text(encoding="utf-8")
    clean = strip_gutenberg(text)
    clean_path.write_text(clean, encoding="utf-8")
    raw_path.unlink()
    print(f"  house_pooh_corner.txt: {len(clean):,} chars")

# Verify total
total = 0
for f in data_dir.glob("*.txt"):
    n = len(f.read_text(encoding="utf-8"))
    total += n
    print(f"  Final: {f.name} = {n:,} chars")
print(f"\n  Total corpus: {total:,} chars")
