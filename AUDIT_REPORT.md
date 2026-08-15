# Applio Fork Audit Report

**Date**: 2026-08-15 (updated after 4-subagent audit + critical fixes)
**Environment**: Kaggle (Python 3.12, torch 2.11.0 cu130) + local CPU test env (Python 3.10, torch 2.13.0+cpu)
**Branch**: `main` @ 817639a7

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
| `_best.pth` tracking | ✅ FIXED | re-extracted on every new low; also epoch 1 |
| Google Drive backup | ✅ NEW | rclone OAuth flow + auto-upload + sync (logic tested with shim) |
| **Legacy weight_norm key load (infer/realtime/onnx)** | ✅ **FIXED** | `remap_weight_norm_keys()`; was the cause of "noise + faint voice" |
| Early-stop final model from best | ✅ FIXED | final checkpoint now uses best EMA weights |
| EMA resume | ✅ FIXED | EMA restored from `_best.pth` on resume |
| DDP deadlock at end of training | ✅ FIXED | barrier before `os._exit(2333333)` |
| Train failure exit code | ✅ FIXED | real failures propagate to the UI |
| HiFi-GAN/old-style pretrains (strict load) | ✅ FIXED | remap applied in pretrain load path |
| resample_sr | ✅ FIXED | output actually resampled before write |
| REST API error codes + temp-dir race | ✅ FIXED | HTTPException + bytes read before tmp cleanup |
| Index path `replace("trained","added")` | ✅ FIXED | removed |
| `index_path=None` crash | ✅ FIXED | guarded |
| Split-audio empty intervals | ✅ FIXED | full-range interval on fallback |
| `app.py --client` NameError | ✅ FIXED | import before use |
| Stop Convert self-kill | ✅ FIXED | never kills own PID, skips non-python PIDs |
| TTS Refresh outputs mismatch | ✅ FIXED | 5 outputs matching `change_choices` |
| Short/empty audio in `change_rms` | ✅ FIXED | guard added |
| Batch conversion bad path | ✅ FIXED | clear error raised |
| reverb `freeze_mode` type | ✅ FIXED | bool() in infer + realtime |
| sid OOB validation | ❌ STILL BROKEN | silent garbage (MEDIUM) |
| `hop_length` param | ❌ STILL IGNORED | dead parameter (LOW) |
| harvest/mangio-crepe | ✅ MITIGATED | removed from UI choices; code branches absent |
| fcpe | ✅ WORKS | model downloads at runtime |
| Secrets in repo | ✅ PASS | no tokens/keys committed |

---

## Subagent audit (2026-08-15): findings by severity

Four research-only subagents covered inference, training, algorithms, and UI/core.

### CRITICAL
1. **Legacy weight_norm keys never remapped on load** — `rvc/infer/infer.py:515`, `rvc/realtime/pipeline.py:75`, `rvc/train/process/export_onnx.py:53` load `strict=False` without remapping `weight_g/weight_v` → every weight-norm tensor silently dropped, decoder runs on **random weights** → garbage/noise output. Source: `extract_model.py:102-110` saves old-style keys. **CONFIRMED manually** (274 old-style keys in base pretrain; merged pretrains are new-style and load fine). **FIXED** + E2E-verified (0 missing/0 unexpected, deterministic infer).
2. **`FileResponse` under `TemporaryDirectory`** — `rvc/infer/infer_api.py:55-88`: response streamed after the dir is torn down → empty/404 downloads. **FIXED** (read bytes inside `with`, `Response` returned after).
3. **`index_path=None` AttributeError** — `rvc/infer/infer.py:290` (`index_path.strip()`). **FIXED** (`(index_path or "")`).
4. **Split-audio fallback leaves `intervals=[]`** — `merge_audio` IndexErrors/garbage. **FIXED** (full-range interval).
5. **DDP deadlock at end of training** — `os._exit(2333333)` only rank 0 → other ranks never exit, parent hangs in `join()`. **FIXED** (final barrier).
6. **HiFi-GAN/RefineGAN pretrains fail strict load** — old-style keys + strict=True → `sys.exit(1)` at start. **FIXED** (remap in pretrain path).
7. **`app.py:324` NameError in `--client` mode** — `infer_api_app` used before import. **FIXED**.

### HIGH
- **EMA not serialized for resume** — resume restarted EMA from raw weights. **FIXED** (restore from `_best.pth`).
- **Best not saved at epoch 1** — `epoch > 1` gate. **FIXED**.
- **Early stop final model = current (worse) weights** — **FIXED** (best EMA swapped in before extraction).
- **Train failures reported as success** — only sentinel exit code checked. **FIXED** (real codes propagate).
- **TTS Refresh 4 outputs vs 5** — `tabs/tts/tts.py:440-443`. **FIXED**.
- **Stop Convert kills the whole app** — `tabs/settings/sections/restart.py:52-74` kills own PID (in-process inference). **FIXED** (self-PID guard + python cmdline check).
- **`resample_sr` still broken** — `infer.py:299-300` changed `tgt_sr` without resampling. **FIXED** (actual resample before write).
- **Index `.replace("trained","added")`** — `infer.py:296`. **FIXED** (removed).

### MEDIUM
- **`change_rms` crash on empty input** — **FIXED**. Batch `os.listdir` without try — **FIXED**.
- **sid OOB** — no range check vs `n_spk` — **UNFIXED**.
- **ZLUDA `z_stft` breaks `torch.stft`** — hijack side effect — **UNFIXED** (low impact: only under ZLUDA).
- **reverb `freeze_mode` passed non-bool** — **FIXED**.
- **`hop_length` ignored** — **UNFIXED** (dead parameter; LOW).

### False positives (manually verified)
- Agent 3 "merge broken, base has no `model` key": **wrong** — `f0G32k.pth` has top-level `['model','iteration','learning_rate']`, 560 tensors; merged pretrains load strict.
- Agent 1 "include_mutes slider value=True": **wrong** — value is already `2`.

---

## Fixed in this session (verified)

1. **Legacy weight_norm key remap** — `rvc/lib/utils.py::remap_weight_norm_keys()` maps `weight_g/weight_v` → `parametrizations.weight.original0/original1`; used in `infer.py`, `realtime/pipeline.py`, `export_onnx.py`, pretrain loading and early-stop best restore. E2E test: full old-style checkpoint (as `extract_model.py` writes) → 0 missing / 0 unexpected, deterministic infer, no NaN.
2. **nvrtc crash** — `rvc/lib/algorithm/commons.py:88`: `@torch.jit.script` replaced with plain tensor ops (identical math, tested fp32/fp16).
3. **torchvision mismatch** — `requirements.txt`: `torchvision==0.26.0`.
4. **Best checkpoint stale** — re-extracted at every new low; now also at epoch 1.
5. **Google Drive backup** — `rvc/lib/tools/gdrive.py` (rclone two-phase OAuth, full-access scope, auto-install, upload_async, sync_logs), UI accordion, Export→Upload outside Colab. **Not runtime-tested in gradio** (no gradio in local env).
6. **BigVGAN/Vocos pretrain merge** — base front-end + NVIDIA decoder, strict-load verified.
7. **Early stop / EMA resume / exit codes / DDP barrier / pretrain remap / API / TTS / restart / split-audio / resample / freeze_mode** — see table above.

## Remaining known issues (unfixed)

### MEDIUM
- **sid OOB** — `rvc/infer/pipeline.py:467`: no range check against `n_spk`; out-of-range sid silently embeds garbage. Add validation.

### LOW
- **`hop_length` parameter ignored** — `rvc/infer/infer.py:219`, `rvc/infer/pipeline.py:434`: never used (F0 uses `self.window`).

## Current training state (user's Kaggle run)

- BigVGAN 24 kHz, merged pretrain, fp16, EMA 0.999, multiscale mel, 200 epochs (22 steps/epoch, ~24 s/epoch).
- Track at epoch ~46: mel 87.2 → 60.5 (displayed, c_mel/3), steady new lows every ~10-20 epochs. Healthy.
- `_best.pth` from this run is stale (pre-fix); use final `my-project_200e_4400s.pth`.
- **The "noise + faint voice" symptom was the inference key-remap bug, not training** — after `git pull`, re-run inference with the trained model to verify.

## Recommendations

1. `git pull` and re-run inference with the trained model — the garbage-audio bug is fixed; this is the first real quality check.
2. Runtime-test the gradio Drive UI once (needs the app on Kaggle).
3. Optional follow-ups: sid OOB validation, real hop_length support, spectral-convergence term, D-pretrains for 40k/48k, `save_every_epoch` lowering (currently 22 steps/epoch means checkpoints every N epochs are sparse).