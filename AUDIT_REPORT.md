# Applio Fork Audit Report

**Date**: 2026-08-15 (updated after training/inference session work)
**Environment**: Kaggle (Python 3.12, torch 2.11.0 cu130) + local CPU test env (Python 3.10, torch 2.13.0+cpu)
**Branch**: `main` @ f565b536

---

## Summary

| Area | Status | Notes |
|------|--------|-------|
| Python syntax (101 tracked .py) | ✅ PASS | All compile |
| Training pipeline (Vocos/BigVGAN) | ✅ PASS | Verified on CPU: learns, fp16 OK, infer OK |
| Vocoder inference duration | ✅ FIXED | hop=256 rework: y_hat == T*hop exactly |
| Deps for Python 3.12 | ✅ FIXED | torch 2.11/0.26.0, silero-vad 5.1.2, click 8.4.2, +8 missing deps |
| nvrtc CUDA kernel crash | ✅ FIXED | `torch.jit.script` removed from fused activation |
| BigVGAN pretrain merge | ✅ FIXED | RVC v2 base front-end + NVIDIA v2 decoder, D = MPD+CQTD |
| EMA generator | ✅ FIXED | decay 0.999, saved to checkpoints |
| `_best.pth` tracking | ✅ FIXED | now re-extracted on every new low |
| Google Drive backup | ✅ NEW | rclone OAuth flow + auto-upload + sync (logic tested with shim) |
| resample_sr | ❌ STILL BROKEN | writes wrong SR without resampling (CRITICAL) |
| REST API error codes | ❌ STILL BROKEN | all errors → 200 (CRITICAL) |
| Index path `replace("trained","added")` | ❌ STILL BROKEN | corrupts valid index paths (HIGH) |
| Short/empty audio in `change_rms` | ❌ STILL BROKEN | crashes on <1s audio (MEDIUM) |
| Batch conversion bad path | ❌ STILL BROKEN | `os.listdir` unhandled (MEDIUM) |
| sid OOB validation | ❌ STILL BROKEN | silent garbage (MEDIUM) |
| `hop_length` param | ❌ STILL IGNORED | dead parameter (LOW) |
| harvest/mangio-crepe | ✅ MITIGATED | removed from UI choices; code branches absent |
| fcpe | ✅ WORKS | model downloads at runtime |
| Secrets in repo | ✅ PASS | no tokens/keys committed |

---

## Fixed in this session (verified)

1. **nvrtc crash** — `rvc/lib/algorithm/commons.py:88`: `@torch.jit.script` replaced with plain tensor ops (identical math, tested fp32/fp16). Scripted kernels JIT-compiled CUDA via nvrtc at first autocast use; `libnvrtc-builtins.so.13.0` missing in Kaggle cu130 env.
2. **torchvision mismatch** — `requirements.txt`: `torchvision==0.26.0` (pairs with torch 2.11.0; preinstalled 0.25.0+cu128 broke transformers 5.x import).
3. **Deps for Python 3.12** — silero-vad 5.1.2, click 8.4.2, added fastapi/httpx/huggingface_hub/local-attention/pandas/regex/resampy/scikit-learn pins.
4. **Best checkpoint stale** — `rvc/train/train.py:1047-1051`: removed `os.path.exists` guard; `_best.pth` now re-extracted at every new lowest loss (was frozen at first low forever).
5. **Google Drive backup** — new `rvc/lib/tools/gdrive.py` (rclone): two-phase OAuth (full-access scope), status check, auto-install, `upload_files`, fire-and-forget `upload_async` in the training loop (G_/D_ every save epoch + best/final, final waits), `sync_logs`; UI accordion in Training tab; Export→Upload now works outside Colab (zip→rclone). Logic tested end-to-end with a fake rclone shim. **Not runtime-tested in gradio** (no gradio in local env).
6. **BigVGAN/Vocos pretrain merge** — `rvc/lib/tools/pretrained_selector.py`: `_merge_base_encoder_flow` (317 tensors from `f0G32k.pth`, old-style weight_norm key mapping), strict-load verified, no HiFiGAN leak, forward smoke OK.
7. **Vocos 24k / BigVGAN 24k training** — verified learn on CPU: mel-only 103→17 in 40 steps; full-loss 276→~20; fp16 no NaN; infer mel 0.65 vs target.

## Remaining known issues (from earlier audit, unfixed)

### CRITICAL
- **`resample_sr` writes wrong SR** — `rvc/infer/infer.py:299-300`: `self.tgt_sr = resample_sr` without resampling; `sf.write(..., self.tgt_sr)` at line 362 → duration stretched (e.g. 16k output = 3x too long). Fix: `librosa.resample(audio_opt, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")` before write.
- **REST API error handling** — `rvc/infer/infer_api.py`: no `HTTPException`; missing file/model, invalid sid/effect, corrupt model all return HTTP 200 with `{"detail": ...}`. Only auth (401) is handled.

### HIGH
- **Index path corruption** — `rvc/infer/infer.py:296`: `.replace("trained", "added")` mangles valid index paths containing "trained". Remove the line.

### MEDIUM
- **Short/empty audio crash** — `rvc/infer/pipeline.py:35-82` (`change_rms`): no guard for empty arrays → `zero-size array to reduction operation maximum`. Add size checks.
- **Batch conversion** — `rvc/infer/infer.py:401`: `os.listdir` unhandled for nonexistent path. Wrap in try/except.
- **sid OOB** — `rvc/infer/pipeline.py:467`: no range check against `n_spk`. Add validation.

### LOW
- **`hop_length` parameter ignored** — `rvc/infer/infer.py:219`, `rvc/infer/pipeline.py:434`: never used (F0 uses `self.window`).

## Current training state (user's Kaggle run)

- BigVGAN 24 kHz, merged pretrain, fp16, EMA 0.999, multiscale mel, 200 epochs (22 steps/epoch, ~24 s/epoch).
- Track at epoch ~46: mel 87.2 → 60.5 (displayed, c_mel/3), steady new lows every ~10-20 epochs. Healthy; final expected ~30-45.
- Note: `_best.pth` from this run is stale (pre-fix); use final `my-project_200e_4400s.pth`.

## Recommendations

1. Fix the 2 CRITICAL inference bugs (resample_sr, API codes) — small, contained changes.
2. Remove index `replace()` line; add the 3 MEDIUM guards.
3. After training completes: listen to final checkpoint; if timbre is right but output dull/noisy, consider multiscale mel tuning or longer training.
4. Runtime-test the gradio Drive UI once (needs the app on Kaggle).