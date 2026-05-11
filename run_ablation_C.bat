@echo off
REM Ablation C: INT8 delta-encoded episode storage vs FP32 baseline
REM Per docs/PER_CHANNEL_ABLATION_PROTOCOL.md
REM Config: 128d / 2 blocks / 30 epochs / BPE / backward pass on / Gutenberg-100
REM Total: 6 runs (3 baseline + 3 variant), ~9 GPU-hours
REM
REM Variant uses --buffer_dtypes episode_values=int8 which triggers symmetric
REM per-tensor INT8 quantization in _store_episode and dequantization in
REM _recall_episode. Per-episode FP32 scale stored in `episode_scales`.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

REM Loads the pre-trained BPE tokenizer (run run_train_tokenizer.bat first)
set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --tokenizer bpe --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --d_model 128 --n_blocks 2 --epochs 30 --stride 64 --backward_pass --output_dir runs/ablation_C

echo ============================================================
echo Ablation C — INT8 episode storage vs FP32 baseline
echo Started: %DATE% %TIME%
echo ============================================================

echo.
echo === BASELINE seed 42 (FP32 episodes) ===
python -m luthi.train %COMMON_ARGS% --seed 42 --run_name baseline_seed42

echo.
echo === BASELINE seed 1337 (FP32 episodes) ===
python -m luthi.train %COMMON_ARGS% --seed 1337 --run_name baseline_seed1337

echo.
echo === BASELINE seed 2026 (FP32 episodes) ===
python -m luthi.train %COMMON_ARGS% --seed 2026 --run_name baseline_seed2026

echo.
echo === VARIANT seed 42 (INT8 episodes) ===
python -m luthi.train %COMMON_ARGS% --seed 42 --run_name variant_int8_episodes_seed42 --buffer_dtypes episode_values=int8

echo.
echo === VARIANT seed 1337 (INT8 episodes) ===
python -m luthi.train %COMMON_ARGS% --seed 1337 --run_name variant_int8_episodes_seed1337 --buffer_dtypes episode_values=int8

echo.
echo === VARIANT seed 2026 (INT8 episodes) ===
python -m luthi.train %COMMON_ARGS% --seed 2026 --run_name variant_int8_episodes_seed2026 --buffer_dtypes episode_values=int8

echo.
echo ============================================================
echo Ablation C complete: %DATE% %TIME%
echo Results in runs/ablation_C/
echo ============================================================
pause
