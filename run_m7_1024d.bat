@echo off
REM M7: v2 1024d / 12 blocks / 1 epoch on the M7 curriculum subset / muPC exp=0.25.
REM
REM Width-scaling validation at production-relevant width following the
REM M6 follow-up decisive result (256d / 12 blocks / 1 epoch on gutenberg_4gb,
REM concluded 2026-05-20, MID-CASE WIN -- v2 substrate trained stably,
REM all blocks active, err_acc halving, no NaN events). M7 jumps to 1024d
REM at the same depth to test whether v2 continues to behave at production-
REM relevant width.
REM
REM Scoping doc: docs/research/2026-05-25_m7-1024d-scoping.md
REM
REM Corpus (revised 2026-05-26): the M7 substrate is no longer trained on
REM the generic gutenberg_4gb corpus. It trains on a curated subset of the
REM 9-stage curriculum:
REM   - Stage 3 (psychology) -- full, ~2.0 GB
REM   - Stage 7 (fantasy) -- full, ~195 MB
REM   - Stage 8 (substack_essays) -- full, ~42 MB
REM   - Stage 1 (science_philosophy) subset: Philosophy, Philosophy_of_Mind,
REM     Consciousness, Neuroscience, Ethics, Logic, Linguistics -- ~1.83 GB
REM Total: 14,752 files, ~4.06 GB. Brian's call 2026-05-26: this version
REM gets no coding knowledge (stage 2 excluded). val_loss is NOT directly
REM comparable to the M6-followup 256d baseline (5.0073) because the corpus
REM has changed -- the substrate-stability hypotheses (H1-H4) are still
REM the load-bearing tests.
REM
REM Held-out probe: ~108 MB stratified sample from mythology_corpus +
REM classics_corpus (both unused stages, both narrative prose). Used for
REM per-epoch held-out perplexity measurement. Trained on different
REM material; in-distribution English narrative.
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
REM   - sparse_threshold not set (off -- sparse PC gating is a separate
REM     planned ablation, would confound width-scaling validation)
REM   - 1 epoch on the M7 curriculum subset (~2.0B BPE tokens, similar
REM     volume to the 256d baseline run)
REM   - val_fraction 0 (100% of curriculum used for training; the held-out
REM     probe is the only val signal -- decided 2026-05-26 with Brian, who
REM     wanted no curriculum content held back from the entity)
REM
REM Estimated model size: ~1.5B params (~580M trainable + ~930M living-
REM weight buffers). Intermediate between current 1024d / 2 v1 (~113M)
REM and the 4B production target.
REM
REM Wall-clock estimate: 5-10 days on DirectML / RX 7800 XT 16GB.
REM
REM What's being tested (full list in scoping doc):
REM   H1 -- Stability at production width (zero NaN, no kernel failures)
REM   H2 -- All 12 blocks remain active (pred_frob climbs in every block)
REM   H3 -- err_acc decreases monotonically across the run
REM   H4 -- muPC exp=0.25 holds at production width
REM   H5 -- (now: held-out probe perplexity decreases; tracks
REM         generalization to in-domain unseen narrative material)
REM
REM Falsification criteria -- stop the run early if any of:
REM   - NaN event in loss or any per-block diagnostic
REM   - pred_frob saturation in deep blocks (last 3-4 blocks plateau)
REM   - err_acc climbing past 0.15 for sustained periods after warmup
REM   - Wall-clock > 14 days without major-milestone progress
REM
REM Tokenizer: corpus_build/tokenizer_32k.json -- 32K BPE trained on the
REM full curriculum corpus (better fit to the M7 corpus than the M6
REM tokenizer, which was trained on gutenberg coverage).
REM
REM Resulting checkpoint at runs/m7_1024d/.../checkpoint.luthi becomes
REM the first v2 substrate at scale that SanctuaryRunner + PCIntensitySource
REM can load. Downstream: turbo threshold tuning per
REM docs/research/2026-05-25_turbo-substrate-mismatch.md.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo M7: v2 1024d / 12blk / 1ep M7-curriculum-subset / muPC exp=0.25
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
  --file-list corpus_build/m7_filelist.txt ^
  --probe-file-list corpus_build/m7_probe_filelist.txt ^
  --val_fraction 0 ^
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
  --run_name v2_1024d_12blocks_1ep_m7corpus_mupc_exp025 ^
  --log-every-batches 100

echo.
echo ============================================================
echo M7 finished: %DATE% %TIME%
echo Output: runs/m7_1024d/v2_1024d_12blocks_1ep_m7corpus_mupc_exp025
echo ============================================================
echo.
echo Next steps:
echo   1. Compare against M6 follow-up (256d / 12) results.json for
echo      H1-H4 (stability metrics) and probe-perplexity trajectory
echo      (corpus differs, so train/val loss are not directly comparable).
echo   2. Load the resulting checkpoint via LuthiModel adapter in
echo      Sanctuary; collect turbo trace via PCIntensitySource for
echo      threshold tuning.
echo   3. If H1-H5 hold, the path to 4096d production scale is open
echo      (gated on optimization stack: Triton kernels, BF16,
echo      sparse PC, ROCm hardware).
