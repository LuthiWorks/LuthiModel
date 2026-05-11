@echo off
REM Phase 4.5a — Tokenizer training + Ablation A only (hard gate before B)
REM Per docs/PER_CHANNEL_ABLATION_PROTOCOL.md and PLAN.md Phase 4.5a.
REM Total: ~11 hours CPU (tokenizer) + ~9 GPU-hours (Ablation A).
REM
REM This wrapper STOPS after Ablation A. Review results in runs/ablation_A/
REM against the protocol pass criteria, then run run_ablation_B.bat manually
REM if A passes. Do not auto-chain to B/C/D.

echo ============================================================
echo Phase 4.5a pipeline: Tokenizer + Ablation A
echo Started: %DATE% %TIME%
echo HARD GATE after A -- review before running B.
echo ============================================================

call "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel\run_train_tokenizer.bat"
if errorlevel 1 (
    echo.
    echo ERROR: tokenizer training failed. Pipeline aborted.
    exit /b 1
)

call "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel\run_ablation_A.bat"

echo.
echo ============================================================
echo Pipeline through Ablation A complete: %DATE% %TIME%
echo NEXT: review runs/ablation_A/ vs protocol pass criteria.
echo If A passes, manually run run_ablation_B.bat. Do not chain.
echo ============================================================
