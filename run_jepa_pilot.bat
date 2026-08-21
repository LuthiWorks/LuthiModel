@echo off
REM Two-arm JEPA pilot -- Experiment 1 (JEPA edition) merged with the M8
REM 256d de-risking pilot (critical-path item 1). 5 seeds per condition.
REM Protocol: docs/research/living-weights-experiments.md (JEPA edition)
REM Criteria: docs/research/2026-07-15_falsification-preregistration.md
REM
REM   run_jepa_pilot.bat 1             -> stage 1: living@256 x5 + dead@256 x5
REM   run_jepa_pilot.bat 2             -> stage 2: dead@{192,384} x5
REM   run_jepa_pilot.bat 3             -> stage 3: dead@512 x5
REM   run_jepa_pilot.bat 1 --dry-run   -> print the plan
REM   run_jepa_pilot.bat --aggregate   -> per-condition summary
REM   run_jepa_pilot.bat 1 --smoke     -> tiny CPU/DML shakeout (minutes)
REM
REM Stage 1 decides half the outcomes alone (living loses/ties at the
REM matched point -> KF2-strong dies, stages 2-3 unnecessary).
REM Resumable: completed runs (pilot_result.json) are skipped.
REM Device: ROCm, enforced (Brian's 2026-08-16 ruling; default declaration
REM since 2026-08-21). The interpreter is the ROCm environment, not the
REM in-repo .venv (CPU-only torch) and not the old Python 3.10 DirectML
REM stack. Override with LUTHI_PYTHON=<path to python.exe>; declare
REM LUTHI_BACKEND=directml|cpu|auto to run off-ROCm deliberately.
REM Mind the game windows: Sunday 2-6 PM, biweekly Friday night.

if "%LUTHI_PYTHON%"=="" set "LUTHI_PYTHON=C:\Dev\rocm-probe\Scripts\python.exe"
if not exist "%LUTHI_PYTHON%" (
    echo [run_jepa_pilot] ROCm interpreter not found: %LUTHI_PYTHON%
    echo [run_jepa_pilot] Set LUTHI_PYTHON to the ROCm python.exe. Refusing to fall back to a bare "python".
    exit /b 2
)

if "%1"=="--aggregate" (
    "%LUTHI_PYTHON%" scripts\jepa_pilot_driver.py --aggregate
) else (
    "%LUTHI_PYTHON%" scripts\jepa_pilot_driver.py --stage %1 %2 %3
)
