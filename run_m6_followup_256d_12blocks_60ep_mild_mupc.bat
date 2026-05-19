@echo off
REM M6 follow-up: v2 256d / 12 blocks / 60 epochs / muPC exponent 0.25.
REM
REM Decisive cross-check: combines all three hypotheses for what might
REM explain M6's depth degradation at 128d into a single run.
REM
REM   - 128d -> 256d: tests whether width-suppression of muPC attenuation
REM     recovers v2 at depth. M5 256d (2 blocks) showed v2 winning 0.64%
REM     vs DeadLM at this width; we need to know whether that holds at
REM     production-relevant depth.
REM   - 20ep -> 60ep: tests under-training. v2 4-block best was at epoch
REM     20 (last) and v2 12-block best was epoch 18 — both indicative of
REM     models still improving when training stopped.
REM   - muPC exponent 0.5 -> 0.25: milder attenuation per block. At L=12
REM     the residual is divided by 1.86 instead of 3.46, preserving more
REM     of the PC signal flow through deeper layers without abandoning
REM     muPC's hyperparameter-transfer property entirely.
REM
REM Single seed (42) per the "fewer seeds, more training" decision.
REM
REM Estimated wall-clock: ~4 days. The big run. Single shot answers all
REM three hypotheses at once -- if val loss recovers to be competitive
REM with the 256d 2-block M5 baseline (~5.72), v2 scales fine at width.
REM If not, the depth-degradation is real and width-independent.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M6 follow-up: v2 256d / 12 blocks / 60 epochs / muPC exp=0.25
echo Started: %DATE% %TIME%
echo ============================================================

python -u -m luthi.v2.m5_runner ^
  --arch v2 ^
  --d_model 256 ^
  --n_blocks 12 ^
  --n_heads 4 ^
  --ffn_expansion 1 ^
  --mu-pc-enabled ^
  --mu-pc-exponent 0.25 ^
  --data_dir corpus_build/gutenberg_100 ^
  --load_tokenizer corpus_build/gutenberg_100_bpe32k.json ^
  --epochs 60 ^
  --batch_size 32 ^
  --seq_len 128 ^
  --stride 64 ^
  --lr 3e-4 ^
  --lr_schedule cosine ^
  --lr_warmup_epochs 2 ^
  --seed 42 ^
  --output_dir runs/m6_followup ^
  --run_name v2_256d_12blocks_60ep_mupc_exp025

echo.
echo ============================================================
echo M6 follow-up finished: %DATE% %TIME%
echo Compare runs/m6_followup/v2_256d_12blocks_60ep_mupc_exp025
echo vs M5 256d v2 baseline (best_val ~5.72 at 2 blocks). If the
echo 12-block run is competitive or better, v2 scales at production
echo width; if not, the depth-degradation is real and load-bearing.
echo ============================================================
