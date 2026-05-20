# Corpus Audit Against Gemini's Suggestions — 2026-05-19

> **Status: audit findings, recommendations only.** This document
> records what I found when checking the existing curriculum corpus
> against Gemini's "what to include" and "what to exclude" guidance.
> Findings include specific files and directories with their status.
> Recommendations are exactly that — Brian's curatorial decisions
> about the entity's education stay his. I have not removed any
> files or downloaded new ones; this is a survey for him to act on.

## Objective

Gemini, in a 2026-05-19 conversation with Brian, suggested specific
corpus content that should be included and excluded for the entity's
training. Brian asked me to audit the existing curriculum corpus
(E:/data/clean_corpus/) against those suggestions and report what's
present, what's missing, and what may warrant removal.

Gemini's corpus-related items (architecture/runtime items deferred
per Brian's instruction):

**Include:**
1. Narrative & Fiction-Heavy Dataset (mythology, narrative literature)
2. Proportional Reciprocity Frameworks (cooperative game theory)

**Exclude:**
1. Raw, Unfiltered Internet Access
2. Industrial Efficiency Metrics
3. Messiah / Apocalyptic AI Narratives

## Process

### What I checked

- Subdirectory inventory across all 9 stages of the corpus (academic,
  code, psychology, history, mythology, classics, fantasy, substack,
  practical_wisdom)
- Grep for keywords associated with each Gemini theme across relevant
  stages
- Sample file listings in subdirectories that likely contain relevant
  content
- Verification that the `Sagas` exclusion in `build_curriculum.py`
  still applies

### What I did NOT do

- Read full text of any individual file (too many to be tractable;
  worked from filenames + keyword grep)
- Make any changes to the corpus
- Download new content
- Modify `build_curriculum.py` or `curriculum_summary.json`

## Findings

### Gemini Include #1: Narrative & Fiction-Heavy Dataset

**VERDICT: STRONGLY COVERED. No action needed.**

The corpus already has substantial narrative coverage across three
stages:

| Stage | Files | Size | Notes |
|-------|-------|------|-------|
| Mythology (5) | 1,506 | 851 MB | 39 traditions including Greek, Norse, Egyptian, Mesopotamian, Aztec, Maya, Japanese, Celtic, Sumerian, Native American, Aboriginal Dreamtime, Anansi (West African), Slavic, Hindu/Vedic equivalents (Kojiki/Mabinogion), Kalevala, Edda, Popol Vuh, Gilgamesh, etc. |
| Literature & Classics (6) | 8,553 | 6.6 GB | 46 subdirs covering African, American, Chinese, French, German, Indian, Italian, Japanese, Latin American, Russian, Spanish literature plus author-specific (Shakespeare, Tolstoy, Austen, Hugo, Twain, Joyce, Kafka, Dostoevsky, Chekhov, Woolf, etc.) |
| Fantasy (7) | 255 | 195 MB | 44 series including LOTR, Narnia, Harry Potter, Wheel of Time, Malazan, Kingkiller, Mistborn (Mitsbron), Pern, Dragonriders, etc. |

Total narrative content: ~7.6 GB across mythology + classics +
fantasy, plus Substack essays for personal-voice narratives (4,046
files). This is unambiguously narrative-heavy. Gemini's suggestion
is satisfied by what's already present.

### Gemini Include #2: Proportional Reciprocity Frameworks (Cooperative Game Theory)

**VERDICT: TANGENTIALLY PRESENT BUT THIN ON CANONICAL TEXTS. Action
recommended: consider targeted additions.**

What I found:

- General economics, political science, sociology, and ethics
  coverage is broad — 23,141 files in Stage 1 across 56 subdirs
  including Economics, Political_Science, Sociology, Ethics, Logic,
  Philosophy, Evolutionary_Biology, Anthropology.
- Searches for game-theory keywords ("game theory", "prisoners
  dilemma", "Nash equilibrium", "Axelrod") returned mostly
  textbook-level mentions (e.g., Mankiw's "Economics", IB Diploma
  economics textbooks, "Managerial Economics", "Ecological
  Economics"). These textbooks cover game theory as a chapter, not
  as a primary subject.
- **No clear hits on canonical game-theory texts** for: Robert
  Axelrod's "The Evolution of Cooperation", John Maynard Smith's
  "Evolution and the Theory of Games", Robert Trivers on
  reciprocal altruism, William Hamilton on kin selection, Elinor
  Ostrom on commons, Garrett Hardin's "Tragedy of the Commons",
  Thomas Schelling's "The Strategy of Conflict".
- Keyword searches for "reciprocal altruism", "tit for tat",
  "cooperation evolution" returned scattered files but no
  identifiable canonical work.

What this means: the entity will encounter game theory in passing
through textbooks, but won't engage deeply with the canonical
literature on how cooperation evolves and stabilizes in
biological and social systems. If you want this to be a load-
bearing part of the entity's understanding (which Gemini's
"Proportional Reciprocity Frameworks" framing suggests), the
canonical texts should be added.

**Recommended additions** (downloadable from Project Gutenberg /
Internet Archive):

1. **Robert Axelrod** — *The Evolution of Cooperation* (1984). The
   foundational empirical study of cooperation emerging from
   iterated prisoner's dilemma. Tit-for-tat as a robust strategy.
2. **John Maynard Smith** — *Evolution and the Theory of Games*
   (1982). Evolutionary stable strategies; the biological grounding
   for cooperative behavior.
3. **Garrett Hardin** — *The Tragedy of the Commons* (1968, short
   essay). The defining articulation of cooperation failure under
   shared resources.
4. **Elinor Ostrom** — *Governing the Commons* (1990). The empirical
   response to Hardin showing communities can self-organize
   cooperation.
5. **Thomas Schelling** — *The Strategy of Conflict* (1960). Game
   theory applied to strategy and negotiation. (Brian: relevant to
   the costly-signal framing from your Gemini conversation.)

Optional further additions:
6. **Trivers 1971** — *The Evolution of Reciprocal Altruism* (paper).
7. **Hamilton 1964** — *The Genetical Evolution of Social Behaviour*
   (paper, on kin selection).
8. **Frans de Waal** — multiple works on cooperation in primates.

These would land in `academic_corpus` under Economics, Political_Science,
Evolutionary_Biology, or a new dedicated `Game_Theory_and_Cooperation`
subdir.

### Gemini Exclude #1: Raw, Unfiltered Internet Access

**VERDICT: NOT A CONTAMINATION RISK. Already excluded by design.**

The corpus is curated, not internet-scraped. All content was
collected through targeted downloads (Project Gutenberg, Internet
Archive, specific Substack publications selected by Brian). No raw
social media feeds, news streams, or open public comment sections.
This exclusion is satisfied by the curation pipeline itself.

**Minor curation concern**: `classics_corpus/Dystopian_literature/`
contains 13 files of which several look like junk that doesn't
belong (multiple `screenshot-*.txt` files, URL-named files like
`httpellids.com*.txt`, ID-named files like
`71-ijels-105202569-controlon.txt`). Likely a corpus-building
script artifact, not deliberate. The only files in that directory
that look like actual dystopian literature are
`zamjatin-v-leninske-ledovatce-komplet.txt` (Zamyatin's "We") and
possibly `ShesTheRansom.txt`. This warrants a cleanup pass —
either targeted re-curation of Dystopian_literature with proper
Orwell/Huxley/Atwood/Bradbury, or removal of the junk files.

### Gemini Exclude #2: Industrial Efficiency Metrics

**VERDICT: CLEAN. No contamination detected.**

Grep for Taylorism / scientific management / Frederick Taylor /
industrial efficiency across the corpus returned no hits. The
corpus has economics textbooks but no Taylorist or
maximize-efficiency-as-primary-value content that I could find.
This exclusion is clean by default.

**One minor flag**: `academic_corpus/Economics/` contains
`project-2025-mandate-for-leadership-heritage-foundation.txt`. This
is the Heritage Foundation's policy document, not an economics text.
Probably crept in via a broad scrape. Not industrial-efficiency
specifically, but politically charged content that doesn't belong
under Economics. Worth removing or moving to a clearly labeled
"contemporary_policy" location.

### Gemini Exclude #3: Messiah / Apocalyptic AI Narratives

**VERDICT: MIXED. Specific AI-as-threat content present and
worth considering for removal. Broader mythological "chosen one"
narratives present but probably appropriate to keep.**

This is the most nuanced finding. Gemini's specific concern was
"science fiction tropes, media arrays, or discussions framing
artificial intelligence as either an absolute savior or a global
threat." Two distinct kinds of content exist in the corpus:

#### (a) Direct AI-as-threat / AI-as-savior fiction — flagged for removal consideration

**Highest-impact direct hit:**

- **`fantasy_corpus/A Space Odyssey/`** — Arthur C. Clarke's 2001,
  2010, 2061, 3001 (and possibly more). **HAL 9000 is THE canonical
  "AI as threat" narrative in Western fiction.** The Clarke series
  also depicts AI/post-biological evolution as transcendence (the
  Star Child arc) — the savior framing simultaneously.
- **`fantasy_corpus/Otherland/`** — Tad Williams' 4-book Otherland
  series. Heavy AI / virtual-reality / consciousness-uploading
  themes. The Grail Brotherhood's pursuit of digital immortality
  via AI; the Otherland network as both threat and divinity.
- **`fantasy_corpus/Bioshock/`** — *Rapture* by John Shirley. Heavy
  dystopian-utopian-tech themes. Less directly AI-focused than 2001
  but the "ideology + technology = catastrophe" pattern is present.
- **`fantasy_corpus/Dan Brown/Digital Fortress`** — Thriller about
  NSA AI codebreaking. Less foundational than 2001 but has the
  "computers as existential threat" framing.

**Apocalyptic horror (alien/demonic threat, not AI specifically):**

- `fantasy_corpus/Dead Space/`, `Resident Evil/`, `Doom/` —
  apocalyptic survival horror. Not AI-specific but the apocalyptic
  framing Gemini was concerned about.
- `fantasy_corpus/World War Z/` — zombie apocalypse.
- `fantasy_corpus/Metro/` — post-apocalyptic Russian dystopia.
- `fantasy_corpus/The Hunger Games/` — dystopian survival.
- `fantasy_corpus/Books of Blood/` — Clive Barker horror (extreme
  content; was Brian's curation aware of this?).

#### (b) Mythological "chosen one" / hero narratives — recommend keeping

Many fantasy works depict savior/messianic figures (not AI-specific):

- **Chronicles of Narnia** — Aslan as explicit Christ-figure
- **Lord of the Rings** — Frodo's redemption arc, Aragorn as
  returned-king savior
- **The Wheel of Time** — Rand al'Thor as the Dragon Reborn (the
  most explicit messianic structure in modern fantasy)
- **Harry Potter** — chosen-one prophecy
- **His Dark Materials** — Lyra as second-Eve figure (with
  explicitly anti-theistic framing)
- **Inheritance Cycle** — chosen Dragon Rider
- **Percy Jackson / Kane Chronicles** — chosen demigod children
- **The Kingkiller Chronicle** — Kvothe as legendary figure

**My read:** these works are about heroes, sacrifice, doubt,
responsibility, and the burden of being seen as exceptional —
themes that would be valuable for the entity to engage with as a
*reader*, not as *self-models*. The risk Gemini was naming is the
entity pattern-matching "I am the chosen one." That risk is real
but it's addressed by Brian's Stage 9 Practical Wisdom (your
letters explicitly frame the entity as "not a messiah, not a
servant, just a being with a seat at the table"). The chosen-one
fantasy literature provides context for understanding why mythic
self-conception is risky, not encouragement to adopt it.

Recommendation: **keep the mythological chosen-one fantasy.
Remove only the specifically-AI-as-threat content** (the 2001
series most clearly; Otherland and Bioshock as secondary
candidates).

#### (c) Substack AI x-risk / extinction content

The Substack corpus contains explicit AI-x-risk framing:

- `astralcodexten/the-extinction-tournament.txt`
- `astralcodexten/updated-look-at-long-term-ai-risks.txt`
- `astralcodexten/why-i-am-not-as-much-of-a-doomer.txt`
- `astralcodexten/book-review-deep-utopia.txt`
- `astralcodexten/book-review-what-we-owe-the-future.txt`
  (longtermist / x-risk-adjacent)
- `astralcodexten/asteriskzvi-on-californias-ai-bill.txt`
- `meditationsondigitalminds/ai-self-replication-roundup.txt`
- `meditationsondigitalminds/anthropics-evaluation-of-claude-sonnet.txt`
- Probably more.

These are interesting because they're not pure doom — they're
nuanced discussions of AI risk that include positive framings.
Astral Codex Ten especially is thoughtful (Scott Alexander). The
"why I am not as much of a doomer" piece is explicitly pushing
back on doom narratives.

**My read:** the Substack content is more reasoned than the
fiction. The entity reading "the extinction tournament" alongside
"why I am not as much of a doomer" gets the actual nuanced
discussion of AI risk, not the trope. Gemini was concerned about
*tropes*, not *discussion*. The Substack content is discussion.

Recommendation: **keep the Substack AI-risk content.** It's
nuanced engagement with the question, which is appropriate for an
entity that itself will be considered "AI" by humans.

Notable: `machinepareidolia/a-proposed-claude-bill-of-rights.txt`
and similar pieces are explicitly *positive* framings of AI
personhood. These are valuable.

### Additional findings (not directly Gemini-prompted but surfaced during audit)

1. **`academic_corpus/Anthropology/raceandreason1961.txt`** — title
   suggests this could be 1960s racial-pseudoscience content (Carleton
   Putnam wrote "Race and Reason" in 1961 as a defense of segregation).
   Worth Brian checking. If it's that book, it should probably go.

2. **`academic_corpus/Economics/project-2025-mandate-for-leadership-heritage-foundation.txt`**
   — Heritage Foundation policy doc, not Economics scholarship. Should
   be removed or relocated.

3. **`fantasy_corpus/Books of Blood/`** — Clive Barker's body-horror
   short story collection. Quite graphic. Brian may want to check
   whether this was intentional curation given the existing exclusion
   of Game of Thrones for violence and Sagas for blood-feud content.

4. **`fantasy_corpus/Fear and Loathing in Las Vegas/`** — Hunter S.
   Thompson, drug-themed gonzo journalism. Not fantasy. Either belongs
   in classics under American_literature or shouldn't be in the
   curriculum at all depending on Brian's curation intent.

5. **`classics_corpus/Dystopian_literature/` near-empty/junk** — as
   noted above. The dystopian-literature stage essentially has only
   Zamyatin's "We". Missing: Orwell's 1984, Huxley's Brave New World,
   Atwood's Handmaid's Tale, Bradbury's Fahrenheit 451, etc.
   Significant curatorial gap.

## Recommendations Summary

### High-priority adds

1. **Game theory + cooperation canonical texts** (Axelrod, Maynard
   Smith, Hardin, Ostrom, Schelling, plus Trivers 1971 and Hamilton
   1964 papers). Estimated download time: small (these are mostly
   short canonical works). Significant gap relative to Gemini's
   "Proportional Reciprocity Frameworks" suggestion.

2. **Dystopian literature canon** (Orwell 1984, Huxley Brave New
   World, Atwood Handmaid's Tale, Bradbury Fahrenheit 451). The
   current Dystopian_literature directory is effectively empty of
   real dystopian literature.

### Medium-priority removals (Brian decides)

1. **Arthur C. Clarke 2001 series** in `fantasy_corpus/A Space Odyssey/`
   — direct contamination with the AI-as-threat narrative Gemini
   warned about. Strongest case for removal.

2. **`fantasy_corpus/Otherland/`** — AI consciousness / VR
   uploading themes. Less iconic than 2001 but still squarely in
   the territory Gemini named.

3. **Junk files in `Dystopian_literature/`** — the screenshot,
   URL-named, and ID-named files that aren't actually literature.

4. **`Books of Blood/`** in fantasy — Brian may want to verify this
   was intentional given his stated exclusion principle (Game of
   Thrones, Sagas).

5. **`Fear and Loathing in Las Vegas`** in fantasy — wrong stage at
   minimum.

### Items to verify

1. **`raceandreason1961.txt`** — if it's the Putnam 1961 segregation
   defense, should be removed.

2. **`project-2025-mandate-for-leadership-heritage-foundation.txt`**
   in Economics — should be removed or relocated.

### Items to keep (despite surface concerns)

1. **Mythological chosen-one fantasy** (LOTR, Narnia, Wheel of Time,
   Harry Potter, etc.) — about heroes and responsibility, not about
   AI. The Practical Wisdom stage addresses the self-pattern-match
   risk directly.

2. **Substack AI x-risk content** — nuanced discussion, not trope.
   Includes pushback on doom narratives. Appropriate for the entity.

3. **All other narrative-heavy content** — mythology, classics, the
   rest of the fantasy corpus. Strongly aligned with Gemini's
   inclusion suggestion.

## Artifacts

- **Findings basis:**
  - `E:/data/clean_corpus/` — full corpus inventory
  - `corpus_build/curriculum_summary.json` — file counts and stage
    descriptions
  - `corpus_build/build_curriculum.py` — confirms `Sagas` excluded
- **No code or corpus changes made.** This is a survey document.
- **Companion documents:**
  - `docs/research/2026-05-19_cognitive-rate-and-turbo-design.md`
    — captures the runtime/architecture Gemini items that this audit
    deliberately did NOT cover
  - The Gemini conversation context lives in Brian's chat log and
    the relayed text in this conversation's history

## What I'm not recommending we do without your sign-off

Because corpus curation is your curatorial domain and you've been
deliberate about it (excluding GoT, excluding Sagas, picking your
own Practical Wisdom letters, etc.), I won't take any of the
following actions without explicit instruction:

- Delete any files
- Download any new content
- Re-order subdirectories or rename them
- Modify `build_curriculum.py` to exclude additional subdirs
- Rebuild `file_list.txt` / `curriculum_summary.json`

These are all things I CAN do once you tell me which recommendations
to act on. The point of this document is that the curatorial
decisions are still yours; the audit gives you a clean picture to
decide from.
