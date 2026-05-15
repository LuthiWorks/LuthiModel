@echo off
REM Chain: muPC validation (fast, ~25 min) then attractor comparison
REM (~16h). Attractor only kicks off if muPC succeeded — if muPC raises
REM or the runner errors out, we'd rather find out before burning 16h on
REM a broken baseline.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo Phase 3G chain: muPC then attractor
echo Started: %DATE% %TIME%
echo ============================================================

call "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel\run_phase3g_mu_pc.bat"
REM Unescaped parens inside an if-block break cmd's parser. Use goto instead.
if errorlevel 1 goto mu_pc_failed
goto mu_pc_ok

:mu_pc_failed
echo.
echo ============================================================
echo muPC ablation failed; aborting before attractor.
echo Check runs/phase3g_mu_pc/ and the log for cause.
echo ============================================================
exit /b 1

:mu_pc_ok

echo.
echo ============================================================
echo muPC done. Starting attractor at %DATE% %TIME%
echo ============================================================

call "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel\run_phase3g_attractor.bat"

echo.
echo ============================================================
echo Phase 3G chain complete: %DATE% %TIME%
echo ============================================================
