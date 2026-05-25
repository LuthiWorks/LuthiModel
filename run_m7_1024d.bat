@echo off
REM M7: v2 1024d / 12 blocks / 1 epoch on gutenberg_4gb / muPC exp=0.25.
REM
REM Width-scaling validation at production-relevant width following the
REM M6 follow-up decisive result (256d / 12 blocks / 1 epoch on gutenberg_4gb,
REM concluded 2026-05-20, MID-CASE WIN — v2 substrate trained stably,
REM all blocks active, err_acc halving, no NaN events). M7 jumps to 1024d
REM at the same depth on the same corpus to test whether v2 continues to
REM behave at production-relevant width.
REM
REM Scoping doc: docs/research/2026-05-25_m7-1024d-scoping.md
REM
REM Configuration choices (Brian-confirmed 2026-05-25):
REM   - d_model 1024 / n_blocks 12 / n_heads 16 (head_dim=64, standard)
REM   - ffn_expansion 1 (matched 256d baseline)
REM   - muPC exp=0.25 (matched 256d; tests whether the exponent generalizes
REM     to production width)
REM   - lr 3e-4 (matched 256d; muPC is designed to preserve lr across
REM     width without re-tuning)
REM   - batch_size 32 (matched 256d; watch first 5 minutes of run for
REM     OOM at the larger d_model, restart with smaller if needed)
REM   - sparse_threshold not set (off — sparse PC gating is a separate
REM     planned ablation, would confound width-scaling validation)
REM   - 1 epoch on gutenberg_4gb (~2.2B BPE tokens, same as 256d baseline)
REM
REM Estimated model size: ~1.5B params (~580M trainable + ~930M living-
REM weight buffers). Intermediate between current 1024d / 2 v1 (~113M)
REM and the 4B production target.
REM
REM Wall-clock estimate: 5-10 days on DirectML / RX 7800 XT 16GB.
REM
REM What's being tested (full list in scoping doc):
REM   H1 — Stability at production width (zero NaN, no kernel failures)
REM   H2 — All 12 blocks remain active (pred_frob climbs in every block)
REM   H3 — err_acc decreases monotonically across the run
REM   H4 — muPC exp=0.25 holds at production width
REM   H5 — Final val_loss outperforms 256d / 12 baseline (5.0073)
REM
REM Falsification criteria — stop the run early if any of:
REM   - NaN event in loss or any per-block diagnostic
REM   - pred_frob saturation in deep blocks (last 3-4 blocks plateau)
REM   - err_acc climbing past 0.15 for sustained periods after warmup
REM   - Wall-clock > 14 days without major-milestone progress
REM
REM Tokenizer: corpus_build/tokenizer_32k.json (same as 256d run, 32K BPE
REM trained for broad corpus coverage). val_loss won't be directly
REM comparable to M5 runs that used the 100-book tokenizer, but is
REM directly comparable to the M6 follow-up 256d / 12 result (same
REM tokenizer, same corpus).
REM
REM Resulting checkpoint at runs/m7_1024d/.../checkpoint.luthi becomes
REM the first v2 substrate at scale that SanctuaryRunner + PCIntensitySource
REM can load. Downstream: turbo threshold tuning per
REM docs/research/2026-05-25_turbo-substrate-mismatch.md.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M7: v2 1024d / 12blk / 1ep gutenberg_4gb / muPC exp=0.25
echo Started: %DATE% %TIME%
echo ============================================================

python -u -m luthi.v2.m5_runner ^
  --arch v2 ^
  --d_model 1024 ^
  --n_blocks 12 ^
  --n_heads 16 ^
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
  --output_dir runs/m7_1024d ^
  --run_name v2_1024d_12blocks_1ep_gutenberg4gb_mupc_exp025 ^
  --log-every-batches 100

echo.
echo ============================================================
echo M7 finished: %DATE% %TIME%
echo Output: runs/m7_1024d/v2_1024d_12blocks_1ep_gutenberg4gb_mupc_exp025
echo ============================================================
echo.
echo Next steps:
echo   1. Compare against M6 follow-up (256d / 12) results.json
echo      for like-for-like val_loss / err_acc / pred_frob trajectory.
echo   2. Load the resulting checkpoint via LuthiModel adapter in
echo      Sanctuary; collect turbo trace via PCIntensitySource for
echo      threshold tuning.
echo   3. If H1-H5 hold, the path to 4096d production scale is open
echo      (gated on optimization stack: Triton kernels, BF16,
echo      sparse PC, ROCm hardware).
