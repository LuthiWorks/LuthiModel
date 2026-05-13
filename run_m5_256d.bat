@echo off
REM M5 re-run at 256d/2 blocks — satisfies the V2 plan §M5 spec
REM (the original M5 pilot ran at 128d/2 to match v1 reference; audit
REM 2026-05-12 correctly flagged this as an undocumented plan deviation).
REM
REM Same 6-run grid as the 128d pilot: 3 seeds × 2 archs.
REM
REM Estimated wall-clock at 256d / Gutenberg-100 / stride=64 / 30 epochs:
REM   v2 runs   : ~12h each (4× the per-flop cost of 128d)
REM   dead runs : ~6h each
REM   Total     : ~54h sequential = ~2.3 days
REM
REM Uses the instrumented m5_runner (audit fix #2) so NFF, prediction
REM Frobenius norm, and precision distribution are captured natively.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M5 re-run at 256d (plan-spec config per V2_IMPLEMENTATION_PLAN.md)
echo Started: %DATE% %TIME%
echo ============================================================

set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 30 --batch_size 32 --seq_len 128 --d_model 256 --n_blocks 2 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 2 --output_dir runs/m5_256d

echo.
echo === v2 seed 42 ===
python -u -m luthi.v2.m5_runner --arch v2 --seed 42 %COMMON_ARGS% --run_name v2_seed42 && ^
echo. && echo === v2 seed 1337 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --seed 1337 %COMMON_ARGS% --run_name v2_seed1337 && ^
echo. && echo === v2 seed 2026 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --seed 2026 %COMMON_ARGS% --run_name v2_seed2026 && ^
echo. && echo === dead seed 42 === && ^
python -u -m luthi.v2.m5_runner --arch dead --seed 42 %COMMON_ARGS% --run_name dead_seed42 && ^
echo. && echo === dead seed 1337 === && ^
python -u -m luthi.v2.m5_runner --arch dead --seed 1337 %COMMON_ARGS% --run_name dead_seed1337 && ^
echo. && echo === dead seed 2026 === && ^
python -u -m luthi.v2.m5_runner --arch dead --seed 2026 %COMMON_ARGS% --run_name dead_seed2026

echo.
echo ============================================================
echo M5 256d re-run finished: %DATE% %TIME%
echo Results in runs/m5_256d/. Next: pytest tests/test_pc_vs_dead.py
echo (after pointing M5_RUNS_DIR to runs/m5_256d, or copying results).
echo ============================================================
