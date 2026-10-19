@echo off
setlocal
REM Talk to the v1-era vision checkpoint.
REM Uses the plaintext copy when it exists (made once by
REM scripts\decrypt_checkpoints.py); otherwise the encrypted .luthi, whose
REM password is read from LUTHI_CHECKPOINT_KEY. Never write a password in
REM this file: the repo is public.
cd /d "%~dp0"
set "CKPT=E:\runs\vision\checkpoint.decrypted.pt"
if not exist "%CKPT%" set "CKPT=E:\runs\vision\checkpoint.luthi"
python -m luthi.generate --checkpoint "%CKPT%" --living --introspect
pause
