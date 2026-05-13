@echo off
REM M6 depth sweep — audit 2026-05-12 follow-up to M5.
REM
REM The 2-block M5 pilot tested exactly 1 inter-block prediction connection.
REM Production target is 36 blocks (35 connections). 4.6's audit flagged
REM "v2 architecturally validated" as overclaiming at 2 blocks — depth
REM behavior is unknown until we sweep.
REM
REM Plan: run v2 at 4 / 8 / 12 blocks with 1 seed each, plus matched DeadLM
REM at the same depths. d_model held at 128 for tractability (256d × 12
REM blocks ≈ 24× M5's per-run cost — too slow for a depth sweep). Single
REM seed because the goal is depth-scaling shape, not per-seed variance —
REM cheaper signal first; if it's stable, full multi-seed at production
REM scale lands in Phase 5.
REM
REM Estimated wall-clock at 128d / Gutenberg-100 / stride=64 / 20 epochs:
REM   v2 4-block  : ~4h    v2 8-block  : ~8h    v2 12-block : ~12h
REM   dead 4-block: ~2h    dead 8-block: ~4h    dead 12-block: ~6h
REM   Total       : ~36h sequential.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M6 depth sweep: v2 PC vs DeadLM at 4/8/12 blocks
echo Started: %DATE% %TIME%
echo ============================================================

set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 20 --batch_size 32 --seq_len 128 --d_model 128 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 2 --seed 42 --output_dir runs/m6_depth

echo.
echo === v2 n_blocks=4 ===
python -u -m luthi.v2.m5_runner --arch v2 --n_blocks 4 %COMMON_ARGS% --run_name v2_4blocks && ^
echo. && echo === v2 n_blocks=8 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --n_blocks 8 %COMMON_ARGS% --run_name v2_8blocks && ^
echo. && echo === v2 n_blocks=12 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --n_blocks 12 %COMMON_ARGS% --run_name v2_12blocks && ^
echo. && echo === dead n_blocks=4 === && ^
python -u -m luthi.v2.m5_runner --arch dead --n_blocks 4 %COMMON_ARGS% --run_name dead_4blocks && ^
echo. && echo === dead n_blocks=8 === && ^
python -u -m luthi.v2.m5_runner --arch dead --n_blocks 8 %COMMON_ARGS% --run_name dead_8blocks && ^
echo. && echo === dead n_blocks=12 === && ^
python -u -m luthi.v2.m5_runner --arch dead --n_blocks 12 %COMMON_ARGS% --run_name dead_12blocks

echo.
echo ============================================================
echo M6 depth sweep finished: %DATE% %TIME%
echo Results in runs/m6_depth/. Plot val curves vs depth to detect
echo cascade-stability issues (v2 should NOT diverge or get worse-than-
echo-DeadLM as depth grows if hierarchical PC dynamics are sound).
echo ============================================================
