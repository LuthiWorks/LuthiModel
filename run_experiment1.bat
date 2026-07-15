@echo off
REM Experiment 1: matched-capacity control sweep (falsification-critical)
REM Protocol: docs/research/living-weights-experiments.md section 2
REM Pre-registration: docs/research/2026-07-15_falsification-preregistration.md
REM 5 seeds per condition (Brian, 2026-07-15). Resumable: completed runs skipped.
REM
REM   run_experiment1.bat 1        -> stage 1: v2@256 x5 + dead@256 x5   (~44h)
REM   run_experiment1.bat 2        -> stage 2: dead@192 x5 + dead@384 x5 (~37h)
REM   run_experiment1.bat 3        -> stage 3: dead@512 x5               (~35h)
REM   run_experiment1.bat 1 --dry-run   -> print the plan without running
REM   run_experiment1.bat --aggregate   -> summarize completed runs
REM
REM Estimates from the M5 256d rerun (~5.5h/v2 run, ~3.3h/dead run).
REM Mind the game windows: Sunday 2-6 PM, biweekly Friday night.

if "%1"=="--aggregate" (
    python scripts\experiment1_driver.py --aggregate
) else (
    python scripts\experiment1_driver.py --stage %1 %2
)
