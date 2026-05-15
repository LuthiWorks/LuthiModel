@echo off
REM Phase 3G ablation #1: Depth-muP / muPC validation at 256d/2 blocks.
REM
REM Source: docs/RESEARCH_LITERATURE_2026-05-13.md (Innocenti et al. 2025).
REM
REM Goal: isolate whether muPC re-parameterization helps or hurts at the
REM pilot scale (2 blocks). If it helps even at L=2, run M6 depth sweep
REM with muPC on. If it hurts, falsifier: do not bundle muPC into M6.
REM
REM Comparison: this v2_seed42 + muPC run vs runs/m5_256d/v2_seed42 (the
REM M5 baseline at the same config minus the muPC flag). 1 epoch is
REM enough to see the init-and-residual-scale effect; full convergence
REM is not required for this ablation.
REM
REM Estimated wall-clock: ~10-15 min at 256d/2 blocks/1 epoch (DirectML).

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo Phase 3G ablation: muPC validation at 256d/2 blocks
echo Started: %DATE% %TIME%
echo ============================================================

set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 1 --batch_size 32 --seq_len 128 --d_model 256 --n_blocks 2 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 0 --seed 42 --output_dir runs/phase3g_mu_pc

echo.
echo === v2 baseline (muPC OFF) ===
python -u -m luthi.v2.m5_runner --arch v2 %COMMON_ARGS% --run_name v2_seed42_no_mu_pc && ^
echo. && echo === v2 with muPC ON === && ^
python -u -m luthi.v2.m5_runner --arch v2 --mu-pc-enabled %COMMON_ARGS% --run_name v2_seed42_mu_pc

echo.
echo ============================================================
echo Phase 3G muPC ablation finished: %DATE% %TIME%
echo Compare runs/phase3g_mu_pc/v2_seed42_no_mu_pc vs
echo         runs/phase3g_mu_pc/v2_seed42_mu_pc on best_val + first-epoch
echo loss trajectory. Falsifier per To-Do.md: convergence penalty >=20%%
echo at L=2 means do not bundle into M6.
echo ============================================================
