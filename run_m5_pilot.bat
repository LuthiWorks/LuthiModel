@echo off
REM M5 head-to-head pilot: v2 PC vs DeadLM baseline.
REM Per docs/V2_IMPLEMENTATION_PLAN.md M5 and docs/V2_PILOT_RESULTS.md.
REM
REM 6 runs total: 3 seeds (42, 1337, 2026) x 2 architectures (v2, dead).
REM Sequential, && between runs so a config-level failure stops the chain
REM rather than burning ~13 hours producing 6 identical errors.
REM
REM Estimated wall-clock at 128d/2/30 epochs/Gutenberg-100/stride=64:
REM   v2 runs   : ~3h each (PC self-mod + consolidation overhead)
REM   dead runs : ~1.5h each (vanilla transformer)
REM   Total     : ~13-14 hours
REM
REM Per-run output: runs/m5/<arch>_seed<seed>/results.json + model_final.
REM After completion: pytest tests/test_pc_vs_dead.py reads results.json
REM files and runs the falsification suite for the M5 verdict.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M5 pilot: v2 PC vs DeadLM head-to-head
echo Started: %DATE% %TIME%
echo ============================================================

REM Shared args between all 6 invocations.
set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 30 --batch_size 32 --seq_len 128 --d_model 128 --n_blocks 2 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 2 --output_dir runs/m5

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
echo M5 pilot finished: %DATE% %TIME%
echo Results in runs/m5/. Next:
echo   pytest tests/test_pc_vs_dead.py
echo ============================================================
