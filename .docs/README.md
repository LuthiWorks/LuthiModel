# .docs — the proof-of-concept record

These documents are the record of the numpy/CPU proof-of-concept phase and
the early v1 (Hebbian) PyTorch work: rich parameters, the v6 trade-off, stress
tests, the 256d scale tests, and the DirectML training log. They are kept as
history, per `CLAUDE.md`, and are not implementation docs.

Read them as what was believed and measured *at the time*. Several of their
findings are recorded in `CLAUDE.md`'s "DO NOT REINVENT" list, some now with
status notes. One of them — "the episode store carries most recall weight" —
does not describe the v2 substrate at scale (see `docs/ARCHITECTURE.md`,
"Memory"). For what the code computes now, see `docs/ARCHITECTURE.md`; for
what is established, see `docs/KEY_FINDINGS.md`.

*Added 2026-10-19.*
