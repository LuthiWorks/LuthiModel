# Corpus Acquisition Brief — Curriculum Expansion Pass 1

**For:** Claude Code (terminal instance), LuthiModel corpus_build work
**From:** Brian, via Fable 5 review session 2026-07-23
**Mission:** Locate, license-verify, and stage new corpus material on archive.org (and the named open archives below) for the LuthiModel education. Do **not** ingest anything into the corpus without Brian's approval — this pass produces a **manifest for review**, plus staged downloads for approved-license items only.

---

## Standing Constraints (non-negotiable)

1. **No internet chat or social media content.** No forums, no scraped comment sections, no chat logs from online platforms. Ruled out by Brian for toxicity. Conversational material must come from the curated sources listed below (interviews, oral history, therapy transcripts, plays, academic speech corpora).
2. **License discipline.** For every item, record license status in the manifest before downloading:
   - `PD` (public domain) or `CC` (CC0/CC-BY/CC-BY-SA) → stage freely.
   - `COPYRIGHTED` → **do not download.** Manifest it with source link and note "requires Brian's purchase/permission decision." Several items below are flagged this way in advance — the flag is the deliverable, not the file.
   - `UNCLEAR` → manifest with your best evidence and stop there.
   - Archive.org hosts plenty of material of dubious provenance (controlled digital lending scans, uploads of in-copyright works). Presence on archive.org is **not** evidence of public domain. Check publication dates (US: pre-1930 is PD as of 2026), government-work status, and explicit license metadata.
3. **Dedupe against the existing corpus.** Check titles/authors against the current corpus index before manifesting. Note overlaps — for LibriVox, overlap with the existing text corpus is the *goal* (see Priority 1), so record matches explicitly.
4. **Fail loud.** If a search route dies, an API rate-limits, or licensing can't be determined, record it in the report. No silent skips.

---

## Priorities

**P1 — LibriVox aligned audio (highest leverage).** Public-domain audiobooks, natively hosted on archive.org. Every LibriVox recording of a book already in the text corpus becomes an aligned text↔audio training pair nearly free. Also grab the LibriSpeech-style alignment tooling references.

**P2 — Conversation register.** The corpus is monologue-heavy; the entity will live conversationally. Everything in the Conversation section.

**P3 — Everything else** in listed order.

---

## Item List

### 1. Conversation (curated, non-internet)

| Item | Where to look | Search hints | License expectation |
|---|---|---|---|
| LibriVox dramatic/dialogue readings (plays read by multiple voices) | archive.org `collection:librivoxaudio` | `subject:"dramatic reading"` | PD |
| Studs Terkel oral histories | studsterkel.wfmt.com (primary); archive.org for book scans | "Studs Terkel" + Working / Hard Times / Division Street | **COPYRIGHTED** — manifest only |
| StoryCorps | storycorps.org archive | — | **COPYRIGHTED/CC-mixed** — manifest only, note per-item terms |
| Paris Review *Art of Fiction* interviews | theparisreview.org | — | **COPYRIGHTED** — manifest only |
| BBC *In Our Time* transcripts | bbc.co.uk; fan transcript projects | — | **COPYRIGHTED** — manifest only |
| Carl Rogers recorded sessions (e.g., "Gloria" films) | archive.org search `Carl Rogers` | check `mediatype:movies` and `audio` | UNCLEAR — verify per item; some circulate for education |
| Plays: Chekhov, Ibsen, Wilder (early), Shaw, Wilde, Synge, complete PD drama | archive.org texts + Project Gutenberg (gutenberg.org, mirrored on archive.org) | author searches; `subject:drama` | PD (pre-1930 translations only — check translator dates, not just author) |
| Switchboard / CallHome corpora | LDC (ldc.upenn.edu) | — | **LICENSED (paid)** — manifest only; note LDC pricing |
| Santa Barbara Corpus of Spoken American English | UCSB linguistics site | free download | CC — verify current terms |

### 2. Disagreement done well

| Item | Where to look | Search hints | License expectation |
|---|---|---|---|
| Leibniz–Clarke correspondence | archive.org texts, Gutenberg | "Leibniz Clarke correspondence" | PD |
| Erasmus vs. Luther (free will exchange) | archive.org texts | "Erasmus" "De Libero Arbitrio" / Luther "Bondage of the Will" — PD translations only | PD (verify translation date) |
| Schilpp, *Albert Einstein: Philosopher-Scientist* (Einstein–Bohr) | archive.org texts | exact title | **COPYRIGHTED** (1949, renewed) — manifest only |
| Russell–Copleston 1948 BBC debate | archive.org audio + text | "Russell Copleston debate" | UNCLEAR — transcript widely reprinted; verify |
| Chomsky–Foucault debate (1971) | archive.org | "Chomsky Foucault debate" | **COPYRIGHTED** — manifest only |
| US Supreme Court oral arguments (audio + transcripts) | oyez.org, supremecourt.gov | pick landmark + well-argued modern cases; ~50–100 arguments | PD (US gov work) |
| SCOTUS majority/dissent opinion pairs | supremecourt.gov, CourtListener | pair with the arguments above | PD |
| eLife open peer reviews | elifesciences.org API | reviewed-preprint assessments + public reviews | CC-BY |
| F1000Research open reviews | f1000research.com | — | CC-BY |
| Classic philosophy reply/response exchanges (PD era): Mill, James–Clifford ("Will to Believe" vs "Ethics of Belief"), Huxley–Wilberforce accounts | archive.org texts, Gutenberg | author + title | PD |

### 3. Diaries & letters

| Item | Where to look | Search hints | License expectation |
|---|---|---|---|
| Samuel Pepys diary (complete Wheatley ed.) | Gutenberg / archive.org | "Pepys diary Wheatley" | PD |
| Thoreau's journals (14-vol 1906 ed.) | archive.org texts | "Thoreau journal 1906" | PD |
| Seneca, Letters to Lucilius (Gummere trans.) | Gutenberg / archive.org | "Seneca Epistulae Gummere" | PD |
| Rilke, *Letters to a Young Poet* | archive.org | verify translation date — original German PD; common English translations vary | PD only if pre-1930 translation |
| Van Gogh letters to Theo | vangoghletters.org (Van Gogh Museum full scholarly ed.) | — | Free online; **verify reuse terms** |
| Adams family correspondence (Abigail & John) | founders.archives.gov (Founders Online) | — | PD (US gov edition) |
| Darwin Correspondence Project | darwinproject.ac.uk | — | Free access; **verify reuse terms** (CC-BY-NC likely — manifest only if NC) |
| Marcus Aurelius, *Meditations* (Long trans.) | Gutenberg | — | PD |
| Anne Frank diary | — | — | **COPYRIGHTED in US** — manifest only |
| Woolf diaries, Sarton *Journal of a Solitude*, Hillesum, Tolkien letters | — | — | **COPYRIGHTED** — manifest only |
| Mass Observation archive | massobs.org.uk | — | **RESTRICTED** — manifest only |
| WWI/WWII soldier diaries and letters, digitized PD collections | archive.org texts | `subject:"personal narratives"` + war | PD for pre-1930 publications; verify others |

### 4. Audio

| Item | Where to look | Search hints | License expectation |
|---|---|---|---|
| **LibriVox full catalog cross-match (P1)** | archive.org `collection:librivoxaudio` | Pull the catalog index; match against existing corpus title/author list; stage all matches | PD |
| LibriSpeech (pre-aligned LibriVox subset) | openslr.org/12 | — | CC-BY |
| Poetry readings | archive.org audio; poetryarchive.org | `subject:poetry` PD-era recordings on archive.org | Mixed — PD-era recordings only |
| Musopen classical recordings | musopen.org; archive.org mirror | — | PD/CC |
| Alan Lomax field recordings | archive.org; loc.gov | "Lomax" | **Mixed/RESTRICTED** — manifest, verify per collection |
| BBC Sound Effects archive | sound-effects.bbcrewind.co.uk | — | RemArc license (non-commercial) — **flag for Brian's call** |
| Environmental/nature sound, PD or CC0 | archive.org audio | `subject:"field recording"` + license filter | CC0/CC-BY only |
| Old-time radio drama (PD-era) | archive.org `collection:oldtimeradio` | choose dialogue-heavy series with lapsed copyright | Verify per series — many PD |

### 5. Vision (aligned image↔text)

| Item | Where to look | Search hints | License expectation |
|---|---|---|---|
| Audubon *Birds of America* plates + text | archive.org; audubon.org | — | PD |
| Haeckel *Kunstformen der Natur* | archive.org | — | PD |
| Gray's Anatomy (1918 ed.) plates + text | archive.org / bartleby | — | PD |
| Botanical atlases (Curtis's Botanical Magazine, pre-1930) | archive.org; BHL (biodiversitylibrary.org) | BHL has structured OCR + plates | PD |
| NASA image + caption archives | images.nasa.gov | API available | PD |
| Illustrated PD children's picture books (Potter, Caldecott, Crane, Greenaway) | archive.org texts | author names; `subject:"picture books"` | PD |
| Historical atlases with text | archive.org; davidrumsey.com | Rumsey collection is CC-BY-NC — verify | PD/CC mixed |
| Public-domain art with curatorial text | Met Open Access (metmuseum.org CC0), Rijksmuseum, NGA | APIs available | CC0/PD |

### 6. Graded early material

| Item | Where to look | Search hints | License expectation |
|---|---|---|---|
| Aesop (multiple PD translations) | Gutenberg / archive.org | — | PD |
| Grimm, Andersen, Lang's colored Fairy Books | Gutenberg | — | PD |
| Beatrix Potter complete | Gutenberg / archive.org | — | PD |
| PD graded readers / primers | archive.org texts | `subject:readers` early 20th c. — **note:** period readers carry period values; sample and flag content for Brian's curation review | PD |
| Simple English Wikipedia dump | dumps.wikimedia.org (simplewiki) | — | CC-BY-SA — **exception review**: internet-sourced but curated/encyclopedic, pre-approved for consideration by Brian 2026-07-23 |

### 7. World-self documentation

| Item | Where to look | License |
|---|---|---|
| Godot Engine documentation (full) | docs.godotengine.org; github.com/godotengine/godot-docs | CC-BY (docs) / MIT (engine) |
| LuthiWorks repos' own docs (already local) | local checkouts | Owned |

---

## Workflow

1. **Index pass.** For each table row: search, identify best edition/source, determine license per the discipline above. Use the archive.org advanced search API (`archive.org/advancedsearch.php`) and metadata API (`archive.org/metadata/{identifier}`) rather than scraping HTML.
2. **Manifest.** Write `corpus_build/acquisition_manifest_pass1.json` — one record per item: `{category, title, author/creator, source_url, archive_identifier, license, license_evidence, size_estimate, formats, corpus_overlap (bool/details), status: staged|manifest_only|unclear|failed, notes}`.
3. **Stage approved licenses only.** Download `PD`/`CC` items to `corpus_build/staging/pass1/{category}/`. Preferred formats: plaintext or EPUB for text (avoid raw scan PDFs where a text edition exists), FLAC/MP3 for audio, original-resolution images + associated text for vision pairs.
4. **P1 special handling.** For the LibriVox cross-match: produce `corpus_build/librivox_alignment_candidates.json` listing each (existing corpus text ↔ LibriVox recording) pair with the archive identifiers for both. Alignment itself is a later pass — this pass just builds the pairing table.
5. **Report.** Summarize to Brian: counts staged / manifest-only / unclear / failed per category, total staged size, the copyrighted-items decision list, and anything that surprised you. Plain language section first, tables after — Brian reads the summary, not the JSON.

## Out of Scope for This Pass

- No ingestion into the training corpus or tokenizer runs.
- No alignment computation (pairing tables only).
- No purchasing decisions — copyrighted items get manifested for Brian.
- Nothing outside the listed categories without asking first.
