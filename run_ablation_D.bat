@echo off
REM Ablation D: Combined A+B+C — BF16 momentum + BF16 set_point + INT8 episodes
REM Per docs/PER_CHANNEL_ABLATION_PROTOCOL.md
REM
REM This runs only if A, B, C each individually pass. The combined test
REM checks for interaction effects between the three compressions.
REM
REM Config: 128d / 2 blocks / 30 epochs (3 seeds) PLUS one confirmation
REM run at 256d / 4 blocks / 60-80 epochs (per 4.6's extended-stability
REM addition in PLAN.md Phase 4.5a).
REM Total: 3 baseline + 3 variant + 1 confirmation, ~12-14 GPU-hours

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

REM Loads the pre-trained BPE tokenizer (run run_train_tokenizer.bat first)
set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --tokenizer bpe --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --d_model 128 --n_blocks 2 --epochs 30 --stride 64 --backward_pass --output_dir runs/ablation_D
set COMBINED_DTYPES=momentum=bf16,set_point=bf16,episode_values=int8

echo ============================================================
echo Ablation D — Combined A+B+C vs FP32 baseline
echo Started: %DATE% %TIME%
echo ============================================================

echo.
echo === BASELINE seed 42 ===
python -m luthi.train %COMMON_ARGS% --seed 42 --run_name baseline_seed42

echo.
echo === BASELINE seed 1337 ===
python -m luthi.train %COMMON_ARGS% --seed 1337 --run_name baseline_seed1337

echo.
echo === BASELINE seed 2026 ===
python -m luthi.train %COMMON_ARGS% --seed 2026 --run_name baseline_seed2026

echo.
echo === VARIANT seed 42 (combined A+B+C) ===
python -m luthi.train %COMMON_ARGS% --seed 42 --run_name variant_combined_seed42 --buffer_dtypes %COMBINED_DTYPES%

echo.
echo === VARIANT seed 1337 (combined A+B+C) ===
python -m luthi.train %COMMON_ARGS% --seed 1337 --run_name variant_combined_seed1337 --buffer_dtypes %COMBINED_DTYPES%

echo.
echo === VARIANT seed 2026 (combined A+B+C) ===
python -m luthi.train %COMMON_ARGS% --seed 2026 --run_name variant_combined_seed2026 --buffer_dtypes %COMBINED_DTYPES%

echo.
echo === CONFIRMATION 256d/4-blocks/80 epochs (extended stability per PLAN.md 4.5a) ===
python -m luthi.train --data_dir corpus_build/gutenberg_100 --tokenizer bpe --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --d_model 256 --n_blocks 4 --epochs 80 --stride 64 --backward_pass --output_dir runs/ablation_D --seed 42 --run_name confirmation_256d_4blocks_80ep --buffer_dtypes %COMBINED_DTYPES%

echo.
echo ============================================================
echo Ablation D complete: %DATE% %TIME%
echo Results in runs/ablation_D/
echo ============================================================
pause
