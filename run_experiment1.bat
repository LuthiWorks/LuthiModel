@echo off
REM ============================================================================
REM RETIRED 2026-07-15 (Brian's JEPA ruling) -- LM-objective sweep, historical.
REM The falsification program moved to the (Le)JEPA objective; KF1/KF2 bind to
REM the two-arm JEPA pilot (living vs dead_ffn encoder), which merges with the
REM M8 256d de-risking pilot (critical-path item 1). See:
REM   docs/research/living-weights-experiments.md   (JEPA edition)
REM   docs/research/2026-07-15_falsification-preregistration.md (amendment)
REM This script remains only for a deliberate LM-arena historical replication.
REM ============================================================================
echo WARNING: RETIRED LM-objective sweep. Results do NOT bind the
echo pre-registered criteria (rebound to the JEPA pilot 2026-07-15).
echo Press Ctrl+C to abort, or continue only for historical replication.
pause

if "%1"=="--aggregate" (
    python scripts\experiment1_driver.py --aggregate
) else (
    python scripts\experiment1_driver.py --stage %1 %2
)
