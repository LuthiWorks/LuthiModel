@echo off
REM run_m8.bat -- launch the M8 multimodal JEPA run and bring up LuthiScope.
REM
REM All arguments pass straight through to the trainer, e.g.:
REM   run_m8.bat --run-dir runs\m8_try2 --lr 5e-5 --seed 7
REM
REM The trainer announces its run dir to LuthiScope (luthi\v2\run_registry.py),
REM so the new run appears in the console -- with a live indicator -- within a
REM couple of seconds. Auto-open lives here, in the launcher, not in the trainer,
REM so headless runs (CI, SSH, cron) stay GUI-free.

REM --- open LuthiScope if it isn't already running (best-effort, detached) ---
tasklist /FI "IMAGENAME eq LuthiScope.exe" 2>NUL | find /I "LuthiScope.exe" >NUL
if errorlevel 1 (
    start "" "C:\Users\Hasha Smokes\Desktop\LuthiWorks\LuthiScope\dist\LuthiScope.exe"
)

REM --- then launch training as usual (every arg forwarded) ---
python luthi\v2\m8_multimodal_smoke.py %*
