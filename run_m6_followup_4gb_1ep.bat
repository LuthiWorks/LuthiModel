@echo off
REM Decisive M6 follow-up: v2 256d / 12 blocks / 1 epoch on gutenberg_4gb / muPC exp=0.25.
REM
REM Single decisive run testing v2's depth-scaling at production-relevant
REM configuration with sufficient data to avoid the overfitting confound.
REM
REM Why this configuration:
REM   - 256d / 12 blocks: production-relevant width and depth
REM   - 1 epoch on gutenberg_4gb (~1B BPE tokens, ~100x gutenberg_100):
REM       gives ~28 tokens/trainable_param (near Chinchilla-optimal). M5's
REM       gutenberg_100 + 60 epochs would have given 60x exposure to 10.6M
REM       tokens = 636M token-views but on a corpus the model could
REM       memorize. 1 epoch on 4gb gives 1B unique token exposures.
REM   - muPC exp=0.25: milder per-block attenuation (residual divided by
REM       1.86 at L=12 instead of 3.46). Tests whether less aggressive
REM       muPC preserves the PC signal flow through deeper layers.
REM   - --log-every-batches 100: per-batch progress streaming. Cadence
REM       calibrated for the long wall-clock (~250K batches expected at
REM       ~1850 tokens/sec; logging every 100 batches gives ~2500 log
REM       lines over ~6 days = ~1 line every 3.5 minutes — visible
REM       progress without log flooding).
REM
REM Wall-clock estimate: ~6 days at 256d / 12 blocks / ~1B tokens.
REM
REM Tokenizer: `tokenizer_32k.json` (April 2026, trained for broader
REM corpus coverage) chosen over `gutenberg_100_bpe32k.json` because the
REM April tokenizer was sized for the multi-thousand-book regime, not
REM just the 100-book subset. Trade-off: val_loss numbers won't be
REM directly comparable to M5 (different tokenizer → different
REM per-token entropy), but the broader corpus needs the broader vocab.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M6 follow-up (decisive): v2 256d / 12blk / 1ep gutenberg_4gb / muPC exp=0.25
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
  --data_dir E:/data/gutenberg_4gb ^
  --load_tokenizer corpus_build/tokenizer_32k.json ^
  --epochs 1 ^
  --batch_size 32 ^
  --seq_len 128 ^
  --stride 64 ^
  --lr 3e-4 ^
  --lr_schedule cosine ^
  --lr_warmup_epochs 0 ^
  --seed 42 ^
  --output_dir runs/m6_followup ^
  --run_name v2_256d_12blocks_1ep_gutenberg4gb_mupc_exp025 ^
  --log-every-batches 100

echo.
echo ============================================================
echo M6 follow-up (decisive) finished: %DATE% %TIME%
echo Compare runs/m6_followup/v2_256d_12blocks_1ep_gutenberg4gb_mupc_exp025
echo against M6's 128d v2_12blocks (best_val 6.46). If 4gb run produces
echo a notably better val_loss curve through the epoch, v2 scales at
echo production-relevant width + corpus. If degradation pattern persists,
echo it's load-bearing evidence v2's substrate has a real depth-scaling
echo issue independent of corpus / training budget.
echo ============================================================
