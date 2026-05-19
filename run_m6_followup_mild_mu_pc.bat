@echo off
REM M6 follow-up #2: v2 12 blocks, μPC exponent 0.25, 30 epochs.
REM
REM Purpose: separate "v2 fails at depth" from "μPC attenuates signal
REM too aggressively at depth." Original μPC uses exponent=0.5 (1/√L).
REM At L=12 that divides per-block signal by 3.46×. M6 showed NFF
REM dropping from 5.77e-3 at L=4 to ~2e-3 at L=12, consistent with the
REM attenuation interpretation. Lowering the exponent to 0.25 gives a
REM milder 1/L^0.25 attenuation (1.86× at L=12 instead of 3.46×) while
REM preserving μPC's hyperparameter-transfer benefit.
REM
REM If val loss + NFF recover at exponent=0.25 relative to the original
REM 12-block run, μPC's attenuation is the load-bearing problem and
REM the production muPC config should be tuned. If results are similar
REM to the original 12-block, μPC isn't the dominant issue.
REM
REM Single seed (42) per the "fewer seeds, more training" decision.
REM 30 epochs to match the original M6 budget so the comparison is clean.
REM
REM Estimated wall-clock: ~12-15 hours at 128d / 12 blocks / 30 epochs.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M6 follow-up #2: v2 12 blocks, muPC exponent 0.25, 30 epochs
echo Started: %DATE% %TIME%
echo ============================================================

python -u -m luthi.v2.m5_runner ^
  --arch v2 ^
  --d_model 128 ^
  --n_blocks 12 ^
  --n_heads 4 ^
  --ffn_expansion 1 ^
  --mu-pc-enabled ^
  --mu-pc-exponent 0.25 ^
  --data_dir corpus_build/gutenberg_100 ^
  --load_tokenizer corpus_build/gutenberg_100_bpe32k.json ^
  --epochs 30 ^
  --batch_size 32 ^
  --seq_len 128 ^
  --stride 64 ^
  --lr 3e-4 ^
  --lr_schedule cosine ^
  --lr_warmup_epochs 2 ^
  --seed 42 ^
  --output_dir runs/m6_followup ^
  --run_name v2_12blocks_mupc_exp025

echo.
echo ============================================================
echo M6 follow-up #2 finished: %DATE% %TIME%
echo Compare runs/m6_followup/v2_12blocks_mupc_exp025 vs
echo         runs/m6_depth/v2_12blocks (exponent 0.5 baseline)
echo on best_val + NFF trajectory.
echo ============================================================
