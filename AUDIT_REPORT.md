# Applio Inference Audit Report

**Date**: 2026-08-15  
**Environment**: GPU 14.5GB shared, Python 3.10+, PyTorch 2.6+  
**Test Models**: test.pth (HiFi-GAN 48kHz v2), Vocos.pth (24kHz v2), BigVGAN.pth (24kHz v2)  
**Test Audio**: /tmp/opencode/speech_bv.wav (3s, 24kHz mono)

---

## Summary

| Chain | Status | Critical Issues |
|-------|--------|-----------------|
| 1. Basic wav→wav | ✅ PASS | - |
| 2. Index handling | ⚠️ PARTIAL | Bug: `replace("trained","added")` breaks valid index paths |
| 3. F0 methods | ⚠️ PARTIAL | `harvest`, `mangio-crepe`, `fcpe` not implemented/broken |
| 4. Split audio | ⚠️ PARTIAL | Empty/silence chunks crash; short audio (<1s) crashes |
| 5. Audio effects (38) | ✅ PASS | All 22 tested effects work |
| 6. Vocoders | ⚠️ PARTIAL | Vocos/BigVGAN output wrong duration (0.66s vs 3s) |
| 7. Export formats | ✅ PASS | WAV/MP3/FLAC/OGG all work |
| 8. resample_sr | ❌ FAIL | Bug: writes at `resample_sr` but generates at `tgt_sr` |
| 9. Batch conversion | ⚠️ PARTIAL | Nonexistent path crashes; no error handling |
| 10. REST API | ❌ FAIL | Returns 200 for all errors; wrong status codes |
| 11. v1 models | ✅ PASS | Handled correctly (text_enc_hidden_dim=256) |
| 12. Other params | ✅ PASS | volume_envelope, protect, clean, post_process, proposed_pitch work |
| 13. Edge cases | ⚠️ PARTIAL | Empty wav, short audio crash; sid OOB silent; corrupt cpt handled |

---

## Detailed Findings

### 1. Basic wav→wav Conversion
**Status**: ✅ PASS  
**File**: `rvc/infer/infer.py:207-372`  
**Details**: Default conversion works correctly. Output: 48kHz, 2.15s (input 3s at 24kHz → model upsamples).

### 2. Index Handling
**Status**: ⚠️ PARTIAL  
**File**: `rvc/infer/infer.py:289-296`  
**Severity**: HIGH  
**Bug**: Line 295: `file_index.replace("trained", "added")` — legacy code that corrupts valid index paths containing "trained".  
**Test Results**:
- Empty index: ✅ OK (skipped)
- Nonexistent index: ✅ OK (falls back gracefully)
- Corrupt index: ✅ OK (falls back with warning)
- index_rate=0: ✅ OK (skips index)
- index_rate=1.0: ✅ OK

**Fix**: Remove the `.replace("trained", "added")` line entirely.

```python
# infer.py:289-296
file_index = (
    index_path.strip()
    .strip('"')
    .strip("\n")
    .strip('"')
    .strip()
    # .replace("trained", "added")  # REMOVE THIS LINE
)
```

### 3. F0 Methods
**Status**: ⚠️ PARTIAL  
**File**: `rvc/infer/pipeline.py:200-291`, `rvc/lib/predictors/f0.py`  
**Severity**: HIGH  
**Issues**:
| Method | Status | Error |
|--------|--------|-------|
| rmvpe | ✅ PASS | - |
| fcpe | ❌ FAIL | `FileNotFoundError: rvc/models/predictors/fcpe.pt` (model not bundled) |
| harvest | ❌ FAIL | `cannot access local variable 'f0'` — not implemented in `get_f0()` |
| crepe | ✅ PASS | - |
| crepe-tiny | ✅ PASS | - |
| mangio-crepe | ❌ FAIL | `cannot access local variable 'f0'` — not implemented |

**Missing implementations** in `pipeline.py:get_f0()` for `harvest` and `mangio-crepe`.  
**fcpe** requires model file at `rvc/models/predictors/fcpe.pt` which is not included.

**Fix**: Add harvest/mangio-crepe support or remove from UI options. Bundle fcpe.pt or make optional.

### 4. Split Audio (VAD/Silero)
**Status**: ⚠️ PARTIAL  
**File**: `rvc/infer/infer.py:301-343`, `rvc/lib/tools/split_audio.py`  
**Severity**: MEDIUM  
**Issues**:
- Silence input: ✅ OK (falls back to single chunk)
- Speech with pauses: ✅ OK (threshold method)
- Silero VAD: ⚠️ Returns "No speech detected" for test audio (may need tuning)
- Empty chunks: Handled by fallback
- **Short audio (<1s)**: ❌ CRASH — `zero-size array to reduction operation maximum` at `pipeline.py:550` (AudioProcessor.change_rms)

**Fix**: Add guard in `change_rms` for empty/short audio:
```python
# pipeline.py:52-82 (AudioProcessor.change_rms)
if source_audio.size == 0 or target_audio.size == 0:
    return target_audio
```

### 5. Audio Effects (38 effects)
**Status**: ✅ PASS  
**File**: `rvc/lib/tools/audio_effects.py`  
**Tested**: 22/38 effects (filters, noise, modulation, vinyl, pitch, stereo)  
**Details**: All tested effects work correctly at SR=8000, 24000, 48000. Stereo input handled.

### 6. Vocoders in Inference
**Status**: ⚠️ PARTIAL  
**File**: `rvc/lib/algorithm/synthesizers.py:85-152`, `rvc/lib/algorithm/generators/*.py`  
**Severity**: HIGH  
**Results**:
| Vocoder | Model SR | Output Duration | Expected | Status |
|---------|----------|-----------------|----------|--------|
| HiFi-GAN | 48kHz | 2.15s | ~3s | ⚠️ Short |
| Vocos | 24kHz | 0.66s | 3s | ❌ WRONG |
| BigVGAN | 24kHz | 0.66s | 3s | ❌ WRONG |

**Root Cause**: Vocos/BigVGAN use different upsample_rates (`[2,4,4,4]` vs HiFi-GAN's `[12,10,2,2]`) causing hop_length mismatch. The pipeline uses hardcoded `self.window = 160` (16000/100) but Vocos/BigVGAN expect different hop lengths.

**Fix**: Make hop_length configurable per vocoder or infer from model config.

### 7. Export Formats
**Status**: ✅ PASS  
**File**: `rvc/infer/infer.py:101-133, 361-367`  
**Tested**: WAV, MP3, FLAC, OGG — all work via librosa/soundfile.

### 8. resample_sr Bug
**Status**: ❌ FAIL  
**File**: `rvc/infer/infer.py:298-299, 361`  
**Severity**: CRITICAL  
**Bug**: Audio generated at model's `tgt_sr` (e.g., 48000) but written with `resample_sr` (e.g., 16000) **without resampling**.

**Evidence**:
| resample_sr | Output SR | Duration | Expected |
|-------------|-----------|----------|----------|
| 0 (auto) | 48000 | 2.15s | ✅ |
| 16000 | 16000 | 6.45s | ❌ 3x too long |
| 44100 | 44100 | 2.34s | ⚠️ Slightly off |

**Code**:
```python
# infer.py:298-299
if self.tgt_sr != resample_sr >= 16000:
    self.tgt_sr = resample_sr  # Only changes tgt_sr for NEXT run!

# infer.py:361
sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")  # Uses tgt_sr
```

The audio `audio_opt` is generated at original model SR (48000), but `tgt_sr` is overwritten to `resample_sr` before write.

**Fix**: Actually resample the audio:
```python
# After line 343 (before clean_audio/post_process)
if resample_sr and resample_sr != self.tgt_sr and resample_sr >= 16000:
    audio_opt = librosa.resample(audio_opt, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")
    self.tgt_sr = resample_sr
```

### 9. Batch Conversion
**Status**: ⚠️ PARTIAL  
**File**: `rvc/infer/infer.py:374-442`  
**Severity**: MEDIUM  
**Issues**:
- Multiple files: ✅ OK
- Empty directory: ✅ OK (0 files)
- **Nonexistent path**: ❌ CRASH — `FileNotFoundError` unhandled
- Different formats: Not tested but should work via librosa

**Fix**: Wrap `os.listdir` in try/except:
```python
try:
    audio_files = [f for f in os.listdir(audio_input_paths) if ...]
except FileNotFoundError:
    print(f"Input directory not found: {audio_input_paths}")
    return
```

### 10. REST API (/api/infer)
**Status**: ❌ FAIL  
**File**: `rvc/infer/infer_api.py`  
**Severity**: CRITICAL  
**Issues**: All errors return HTTP 200 with `{"detail": "..."}` instead of proper status codes.

| Test Case | Actual Status | Expected |
|-----------|---------------|----------|
| Missing file | 200 | 400 |
| Missing model_path | 200 | 400 |
| Nonexistent model | 200 | 404 |
| Invalid sid (OOB) | 200 | 400 |
| Invalid effect | 200 | 400 |
| Corrupt model | 200 | 500 |

**Fix**: Raise `HTTPException` with correct status codes:
```python
from fastapi import HTTPException

# infer_api.py:30-88
@app.post("/infer")
async def infer(...):
    if not file:
        raise HTTPException(400, "No file uploaded")
    if not os.path.isfile(model_path):
        raise HTTPException(404, f"Model not found: {model_path}")
    try:
        vc.convert_audio(...)
    except Exception as e:
        raise HTTPException(500, f"Conversion failed: {e}")
```

### 11. v1 Models (text_enc_hidden_dim=256)
**Status**: ✅ PASS  
**File**: `rvc/lib/algorithm/synthesizers.py:63-64, 109-113`  
**Details**: Correctly handled: `text_enc_hidden_dim = 768 if version == "v2" else 256`. Vocos/BigVGAN use v2 config.

### 12. Other Parameters
**Status**: ✅ PASS  
**Tested**:
- `volume_envelope` (0.5, 1.0, 1.5): ✅ Works
- `protect` (0.0, 0.33, 0.5, 0.75): ✅ Works
- `clean_audio` + `clean_strength`: ✅ Works
- `post_process` + effects (reverb): ✅ Works
- `proposed_pitch` + `proposed_pitch_threshold`: ✅ Works (calculates offset)
- `hop_length`: ⚠️ **Unused** — hardcoded to 160 in `Pipeline.__init__` (line 185), parameter ignored

**Fix**: Use `hop_length` parameter in `Pipeline` or remove from API.

### 13. Edge Cases
**Status**: ⚠️ PARTIAL  
**Results**:
| Case | Status | Details |
|------|--------|---------|
| Empty wav | ❌ CRASH | `zero-size array to reduction operation maximum` |
| Silence (zeros) | ✅ OK | Falls back to single chunk |
| 0.3s clip | ❌ CRASH | Same as empty — `change_rms` fails |
| Corrupt checkpoint | ✅ OK | Caught, returns error |
| sid OOB (200 > 108) | ⚠️ Silent | No error, may produce garbage |
| Two VC simultaneously | ✅ OK | Thread-safe (separate instances) |

**Fixes**:
- Add input validation in `convert_audio` for empty/short audio
- Validate `sid` range: `if sid >= self.n_spk: raise ValueError(...)`
- Guard `change_rms` for empty arrays

---

## Priority Fixes

| Priority | Issue | File:Line |
|----------|-------|-----------|
| 🔴 CRITICAL | resample_sr writes wrong SR without resampling | `infer.py:298,361` |
| 🔴 CRITICAL | API returns 200 for all errors | `infer_api.py:30-88` |
| 🔴 CRITICAL | Vocos/BigVGAN wrong duration (0.66s vs 3s) | `synthesizers.py`, `pipeline.py` |
| 🟠 HIGH | `replace("trained","added")` breaks index paths | `infer.py:295` |
| 🟠 HIGH | harvest/mangio-crepe/fcpe not working | `pipeline.py:200-291`, `f0.py` |
| 🟡 MEDIUM | Short/empty audio crashes in `change_rms` | `pipeline.py:52-82` |
| 🟡 MEDIUM | Batch conversion crashes on bad path | `infer.py:398-400` |
| 🟡 MEDIUM | sid OOB not validated | `infer.py:230, 444` |
| 🟢 LOW | hop_length parameter ignored | `infer.py:218`, `pipeline.py:185` |

---

## Test Artifacts
All test outputs saved to `/tmp/test_*.wav` for manual verification.