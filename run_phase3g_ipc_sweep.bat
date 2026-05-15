@echo off
REM Phase 3G ablation #2: iPC T-sweep at 256d/2 blocks.
REM
REM Source: docs/RESEARCH_LITERATURE_2026-05-13.md (Salvatori et al. 2024,
REM "Incremental Predictive Coding").
REM
REM Goal: measure (a) val loss at matched external-forward count, (b)
REM wall-clock per epoch as T grows. The iPC paper claims T=3-5 converges
REM faster per external forward; we expect ~1.5-2x total compute trade
REM for faster convergence. Sweep T in {1, 3, 5}.
REM
REM T=1 baseline run is technically redundant with runs/m5_256d/v2_seed42
REM but we re-run here so wall-clock numbers are apples-to-apples (the
REM iPC inner-loop overhead can shift even at T=1 due to extra branching).
REM
REM Estimated wall-clock: T=1 ~5.5h, T=3 ~10h, T=5 ~16h.
REM Total: ~32h sequential. Consider running overnight.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo Phase 3G ablation: iPC T-sweep at 256d/2 blocks
echo Started: %DATE% %TIME%
echo ============================================================

set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 30 --batch_size 32 --seq_len 128 --d_model 256 --n_blocks 2 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 2 --seed 42 --output_dir runs/phase3g_ipc

echo.
echo === iPC T=1 (classical PC baseline) ===
python -u -m luthi.v2.m5_runner --arch v2 --inference-steps-per-forward 1 %COMMON_ARGS% --run_name v2_seed42_T1 && ^
echo. && echo === iPC T=3 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --inference-steps-per-forward 3 %COMMON_ARGS% --run_name v2_seed42_T3 && ^
echo. && echo === iPC T=5 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --inference-steps-per-forward 5 %COMMON_ARGS% --run_name v2_seed42_T5

echo.
echo ============================================================
echo Phase 3G iPC sweep finished: %DATE% %TIME%
echo Falsifier per To-Do.md: T=5 must beat T=1 by >=10%% val loss at
echo matched external-forward count, OR iPC + grad-checkpoint must be
echo made compatible at the architecture level.
echo ============================================================
