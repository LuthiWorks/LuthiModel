@echo off
REM M6 follow-up #1: v2 12 blocks, μPC default, 60 epochs.
REM
REM Purpose: separate "v2 fails at depth" from "deeper models need more
REM training to converge." The original M6 12-block run was 20 epochs;
REM val loss plateaued ~6.71 and best epoch was early (suggesting
REM possible under-training, not failure). 60 epochs is 3× the original
REM budget at the same depth/width/config.
REM
REM If val loss recovers to be competitive with the v2 4-block result
REM (~5.94) at 60 epochs, the M6 verdict was about epoch budget, not
REM about v2 scaling. If val loss stays plateaued near 6.7, then more
REM training isn't the answer and we look to the μPC-exponent
REM intervention next.
REM
REM Single seed (42) per the "fewer seeds, more training" decision.
REM
REM Estimated wall-clock: ~12-15 hours at 128d / 12 blocks / 60 epochs.
REM Same config as the original M6 12-block run except --epochs.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M6 follow-up #1: v2 12 blocks, muPC default, 60 epochs
echo Started: %DATE% %TIME%
echo ============================================================

python -u -m luthi.v2.m5_runner ^
  --arch v2 ^
  --d_model 128 ^
  --n_blocks 12 ^
  --n_heads 4 ^
  --ffn_expansion 1 ^
  --mu-pc-enabled ^
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
  --run_name v2_12blocks_60ep

echo.
echo ============================================================
echo M6 follow-up #1 finished: %DATE% %TIME%
echo Compare runs/m6_followup/v2_12blocks_60ep vs
echo         runs/m6_depth/v2_12blocks on best_val + final NFF.
echo If 60ep result is competitive with 4-block run, M6's verdict
echo was about epoch budget, not v2 scaling.
echo ============================================================
