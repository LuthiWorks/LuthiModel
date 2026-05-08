@echo off
REM One-time BPE tokenizer training for the ablation pipeline.
REM Trains a 32K-vocab BPE on Gutenberg-100, saves to a stable path,
REM then exits before model training. All subsequent ablation runs
REM (A/B/C/D) load this tokenizer via --load_tokenizer instead of
REM retraining BPE for every job.
REM
REM Cost: ~11 hours, CPU-bound. Run before any of the run_ablation_*.bat
REM scripts. Total pipeline savings vs untrained ablation runs: ~253 hours.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo One-time BPE tokenizer training
echo Started: %DATE% %TIME%
echo Saving to: corpus_build/gutenberg_100_bpe32k.json
echo ============================================================

python -m luthi.train ^
    --data_dir corpus_build/gutenberg_100 ^
    --tokenizer bpe ^
    --bpe_vocab_size 32000 ^
    --bpe_only ^
    --save_tokenizer corpus_build/gutenberg_100_bpe32k.json ^
    --output_dir runs/_tokenizer_only ^
    --run_name bpe32k

echo.
echo ============================================================
echo Tokenizer training complete: %DATE% %TIME%
echo Saved to: corpus_build/gutenberg_100_bpe32k.json
echo Now run run_ablation_A.bat (etc.) — they load this tokenizer.
echo ============================================================
pause
