@echo off
REM Phase 3G ablation #4: Salvatori attractor consolidation comparison at 256d/2 blocks.
REM
REM Source: docs/RESEARCH_SALVATORI_ATTRACTOR_MEMORY.md (Salvatori et al. 2023,
REM "Associative Memories via Predictive Coding"). Implementation landed
REM 2026-05-14 in luthi/v2/consolidation.py.
REM
REM Goal: empirically compare the three consolidation pathways:
REM   - gradient    : pull weight toward stored snapshot (M5 baseline)
REM   - attractor   : replay stored input pattern through pc_self_modify
REM   - both        : gradient first, then attractor (additive)
REM
REM No falsifier as such — Brian's 2026-05-14 design call was that
REM attractor dynamics are worth having on their own merits, not as a
REM remedy for a deficit. This ablation measures *how much* attractor
REM adds, not whether to keep it.
REM
REM Estimated wall-clock: ~5.5h per run x 3 runs = ~16h sequential.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo Phase 3G ablation: consolidation-style comparison at 256d/2 blocks
echo Started: %DATE% %TIME%
echo ============================================================

set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 30 --batch_size 32 --seq_len 128 --d_model 256 --n_blocks 2 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 2 --seed 42 --output_dir runs/phase3g_attractor

echo.
echo === consolidation-style = gradient (M5 baseline) ===
python -u -m luthi.v2.m5_runner --arch v2 --consolidation-style gradient %COMMON_ARGS% --run_name v2_seed42_gradient && ^
echo. && echo === consolidation-style = attractor === && ^
python -u -m luthi.v2.m5_runner --arch v2 --consolidation-style attractor --consolidation-attractor-passes 1 %COMMON_ARGS% --run_name v2_seed42_attractor && ^
echo. && echo === consolidation-style = both === && ^
python -u -m luthi.v2.m5_runner --arch v2 --consolidation-style both --consolidation-attractor-passes 1 %COMMON_ARGS% --run_name v2_seed42_both

echo.
echo ============================================================
echo Phase 3G attractor ablation finished: %DATE% %TIME%
echo Compare best_val + NFF trajectory across the three runs. The
echo behavioral signature (partial-cue recall, perturbation robustness)
echo lives in the catastrophic-forgetting harness, not in val loss alone.
echo ============================================================
