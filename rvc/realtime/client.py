import os
import sys
import json
from fastapi import FastAPI, WebSocketDisconnect, WebSocket, Request
import numpy as np
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from .core import VoiceChanger, AUDIO_SAMPLE_RATE

from assets.auth import check_api_token

app = FastAPI()
vc_instance = None
params = {}

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_ACTIVE_WS = 4
_active_ws = 0


def _ws_authenticated(ws) -> bool:
    token = ws.query_params.get("token", "")
    return check_api_token(token)


@app.middleware("http")
async def api_auth_middleware(request, call_next):
    from fastapi.responses import JSONResponse

    token = request.headers.get("Authorization", "")
    if token.startswith("Bearer "):
        token = token[len("Bearer ") :]
    if not check_api_token(token):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.websocket("/change-config")
async def change_config(ws: WebSocket):
    global vc_instance, params

    await ws.accept()

    if not _ws_authenticated(ws):
        await ws.close(code=4401)
        return

    if vc_instance is None:
        await ws.close(code=4404)
        return

    text = await ws.receive_text()
    jsons = json.loads(text)

    if jsons["if_kwargs"] and jsons["value"] is not None:
        params.setdefault("kwargs", {})[jsons["key"]] = jsons["value"]
    elif jsons["value"] is not None:
        params[jsons["key"]] = jsons["value"]

    crossfade_frame = int(
        max(0.0, min(float(params.get("cross_fade_overlap_size", 0.1)), 2.0))
        * AUDIO_SAMPLE_RATE
    )
    extra_frame = int(
        max(0.0, min(float(params.get("extra_convert_size", 0.5)), 4.0))
        * AUDIO_SAMPLE_RATE
    )

    if (
        vc_instance.crossfade_frame != crossfade_frame
        or vc_instance.extra_frame != extra_frame
    ):
        # Deleting these things is not a good idea; they should only be overwritten directly.
        # del (
        #     vc_instance.vc_model.audio_buffer,
        #     vc_instance.vc_model.convert_buffer,
        #     vc_instance.vc_model.pitch_buffer,
        #     vc_instance.vc_model.pitchf_buffer,
        # )
        del (
            vc_instance.fade_in_window,
            vc_instance.fade_out_window,
            vc_instance.sola_buffer,
        )

        vc_instance.vc_model.realloc(
            vc_instance.block_frame,
            vc_instance.extra_frame,
            vc_instance.crossfade_frame,
            vc_instance.sola_search_frame,
        )
        vc_instance.generate_strength()

    vc_instance.vc_model.input_sensitivity = 10 ** (
        params.get("silent_threshold", -90) / 20
    )

    vad_enabled = params.get("vad_enabled", True)
    if vad_enabled is False:
        vc_instance.vc_model.vad = None
    elif vad_enabled and vc_instance.vc_model.vad is None:
        from rvc.realtime.utils.vad import VADProcessor

        vc_instance.vc_model.vad = VADProcessor(
            sensitivity_mode=3,
            sample_rate=vc_instance.vc_model.sample_rate,
            frame_duration_ms=30,
        )

    # The VAD parameters have been assigned by default.
    # if vc_instance.vc_model.vad is not None:
    #     vc_instance.vc_model.vad.vad.set_mode(vad_sensitivity)
    #     vc_instance.vc_model.vad.frame_length = int(vc_instance.vc_model.sample_rate * (vad_frame_ms / 1000.0))

    clean_audio = params.get("clean_audio", False)
    clean_strength = params.get("clean_strength", 0.5)

    if clean_audio is False:
        vc_instance.vc_model.reduced_noise = None
    elif clean_audio and vc_instance.vc_model.reduced_noise is None:
        from noisereduce.torchgate import TorchGate

        vc_instance.vc_model.reduced_noise = TorchGate(
            vc_instance.vc_model.pipeline.tgt_sr,
            prop_decrease=clean_strength,
        ).to(vc_instance.vc_model.device)

    if vc_instance.vc_model.reduced_noise is not None:
        vc_instance.vc_model.reduced_noise.prop_decrease = clean_strength

    post_process = params.get("post_process", False)
    kwargs = params.get("kwargs", {})

    if post_process is False:
        vc_instance.vc_model.board = None
        vc_instance.vc_model.kwargs = None
    elif post_process and vc_instance.vc_model.kwargs != kwargs:
        # Post-process requires creating a new pendalboard.
        new_board = vc_instance.vc_model.setup_pedalboard(**kwargs)
        vc_instance.vc_model.board = new_board
        vc_instance.vc_model.kwargs = kwargs.copy()

    model_pth = params.get("model_path", vc_instance.vc_model.model_path)
    if model_pth and vc_instance.vc_model.model_path != model_pth:
        import asyncio
        import torch
        import torchaudio.transforms as tat

        from rvc.lib.utils import validate_ui_path

        model_pth = validate_ui_path(model_pth)

        def _swap_model():
            vc_instance.vc_model.model_path = model_pth
            vc_instance.vc_model.pipeline.vc.load_model(model_pth)
            vc_instance.vc_model.pipeline.vc.setup_network()
            # Set a new version, otherwise it will crash.
            vc_instance.vc_model.pipeline.version = (
                vc_instance.vc_model.pipeline.vc.version
            )
            vc_instance.vc_model.pipeline.use_f0 = vc_instance.vc_model.pipeline.vc.use_f0
            vc_instance.vc_model.pipeline.tgt_sr = vc_instance.vc_model.pipeline.vc.tgt_sr

            vc_instance.vc_model.resample_out = tat.Resample(
                orig_freq=vc_instance.vc_model.pipeline.tgt_sr,
                new_freq=AUDIO_SAMPLE_RATE,
                dtype=torch.float32,
            ).to(vc_instance.vc_model.device)

        await asyncio.get_running_loop().run_in_executor(None, _swap_model)

        if clean_audio:
            from noisereduce.torchgate import TorchGate

            vc_instance.vc_model.reduced_noise = TorchGate(
                vc_instance.vc_model.pipeline.tgt_sr,
                prop_decrease=clean_strength,
            ).to(vc_instance.vc_model.device)

    sid = params.get("sid", vc_instance.vc_model.pipeline.sid)
    if vc_instance.vc_model.pipeline.sid != sid:
        import torch

        # This is for multi-SID models.
        vc_instance.vc_model.pipeline.torch_sid = torch.tensor(
            [sid], device=vc_instance.vc_model.pipeline.device, dtype=torch.int64
        )

    index_path = params.get("index_path", None)
    if index_path:
        if vc_instance.vc_model.index_path != index_path:
            from rvc.lib.utils import validate_ui_path

            index_path = validate_ui_path(index_path)
            from rvc.realtime.utils.torch import IndexWrapper

            try:
                index = IndexWrapper(
                    index_path.strip()
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip()
                    .replace("trained", "added"),
                    device=vc_instance.device,
                    dtype=vc_instance.vc_model.dtype,
                )
                big_tsr, _ = index.read_index_tensor()

                vc_instance.vc_model.pipeline.index = index
                vc_instance.vc_model.pipeline.big_tsr = big_tsr
                vc_instance.vc_model.index_path = index_path
            except Exception as error:
                print(f"Failed to load index {index_path}: {error}")
                vc_instance.vc_model.pipeline.index = None
                vc_instance.vc_model.pipeline.big_tsr = None
                vc_instance.vc_model.index_path = None
    else:
        vc_instance.vc_model.pipeline.index = None
        vc_instance.vc_model.pipeline.big_tsr = None
        vc_instance.vc_model.index_path = None

    f0_method = params.get("f0_method", vc_instance.vc_model.pipeline.f0_method)
    if vc_instance.vc_model.pipeline.f0_method != f0_method:
        f0_model = vc_instance.vc_model.pipeline.setup_f0(f0_method)
        vc_instance.vc_model.pipeline.f0_model = f0_model
        vc_instance.vc_model.pipeline.f0_method = f0_method

    embedder_model = params.get("embedder_model", vc_instance.vc_model.embedder_model)
    embedder_model_custom = params.get(
        "embedder_model_custom", vc_instance.vc_model.embedder_model_custom
    )

    if (
        vc_instance.vc_model.embedder_model != embedder_model
        or vc_instance.vc_model.embedder_model_custom != embedder_model_custom
    ):
        old_hubert_model = vc_instance.vc_model.pipeline.hubert_model
        del old_hubert_model

        from rvc.lib.utils import load_embedding, validate_ui_path

        if embedder_model_custom:
            embedder_model_custom = validate_ui_path(embedder_model_custom)

        hubert_model = load_embedding(embedder_model, embedder_model_custom)
        hubert_model = hubert_model.to(vc_instance.device).float()
        hubert_model.eval()

        vc_instance.vc_model.pipeline.hubert_model = hubert_model
        vc_instance.vc_model.embedder_model = embedder_model
        vc_instance.vc_model.embedder_model_custom = embedder_model_custom


@app.post("/record")
async def record(request: Request):
    global vc_instance

    if request.headers.get("content-length"):
        try:
            if int(request.headers["content-length"]) > MAX_AUDIO_BYTES:
                return {"type": "error", "value": "Request body too large"}
        except ValueError:
            pass

    data = await request.json()
    record_button = data.get("record_button", "Stop")
    record_audio_path = data.get("record_audio_path", None)
    export_format = data.get("export_format", "WAV")

    if vc_instance is None:
        return {
            "type": "warnings",
            "value": "Realtime pipeline not found!",
            "button": "Start",
            "path": None,
        }

    if record_button == "Start":
        if not record_audio_path:
            record_audio_path = os.path.join(
                now_dir, "assets", "audios", "record_audio.wav"
            )
        else:
            from rvc.lib.utils import ensure_within_root

            record_audio_path = ensure_within_root(record_audio_path, now_dir)

        vc_instance.record_audio = True
        vc_instance.record_audio_path = record_audio_path
        vc_instance.export_format = export_format
        vc_instance.setup_soundfile_record()

        return {
            "type": "info",
            "value": "Start recording...",
            "button": "Stop",
            "path": None,
        }
    else:
        vc_instance.record_audio = False
        vc_instance.record_audio_path = None
        vc_instance.soundfile = None

        return {
            "type": "info",
            "value": "Stop recording!",
            "button": "Start",
            "path": record_audio_path,
        }


@app.websocket("/ws-audio")
async def websocket_audio(ws: WebSocket):
    global vc_instance, params, _active_ws
    await ws.accept()

    if not _ws_authenticated(ws):
        await ws.close(code=4401)
        return

    if _active_ws >= MAX_ACTIVE_WS:
        await ws.close(code=4403, reason="Too many connections")
        return
    _active_ws += 1

    print("[WS] Connected!")

    try:
        text = await ws.receive_text()
        params = json.loads(text)

        block_frame = max(1, min(int(params["block_frame"]), AUDIO_SAMPLE_RATE))

        print("Starting Realtime...")

        if vc_instance is None:
            import asyncio

            vc_instance = await asyncio.to_thread(
                VoiceChanger,
                block_frame=block_frame,
                cross_fade_overlap_size=max(
                    0.0, min(float(params["cross_fade_overlap_size"]), 2.0)
                ),
                extra_convert_size=max(0.0, min(float(params["extra_convert_size"]), 4.0)),
                model_path=params["model_path"],
                index_path=str(params["index_path"]),
                f0_method=params["f0_method"],
                embedder_model=params["embedder_model"],
                embedder_model_custom=params["embedder_model_custom"],
                silent_threshold=params["silent_threshold"],
                vad_enabled=params["vad_enabled"],
                vad_sensitivity=3,
                vad_frame_ms=30,
                sid=params["sid"],
                clean_audio=params["clean_audio"],
                clean_strength=params["clean_strength"],
                post_process=params["post_process"],
                **params["kwargs"]
            )

        print("Realtime is ready!")

        import asyncio

        loop = asyncio.get_running_loop()

        while True:
            audio = await ws.receive_bytes()
            if len(audio) > MAX_AUDIO_BYTES:
                continue
            arr = np.frombuffer(audio, dtype=np.float32)

            if arr.size != block_frame:
                arr = (
                    np.pad(arr, (0, block_frame - arr.size)).astype(np.float32)
                    if arr.size < block_frame
                    else arr[:block_frame].astype(np.float32)
                )

            if vc_instance is None:
                # Avoid errors when disconnecting.
                return

            audio_output, vol, perf = await loop.run_in_executor(
                None,
                vc_instance.on_request,
                arr * (params["input_audio_gain"] / 100.0),
                f0_up_key=params["f0_up_key"],
                index_rate=params["index_rate"],
                protect=params["protect"],
                volume_envelope=params["volume_envelope"],
                f0_autotune=params["autotune"],
                f0_autotune_strength=params["autotune_strength"],
                proposed_pitch=params["proposed_pitch"],
                proposed_pitch_threshold=params["proposed_pitch_threshold"],
            )

            await ws.send_text(
                json.dumps({"type": "latency", "value": perf[1], "volume": vol})
            )
            await ws.send_bytes(audio_output.tobytes())
    except WebSocketDisconnect:
        print("[WS] Disconnected!")
    except Exception as error:
        print(f"[WS] Error: {error}")
    finally:
        _active_ws = max(0, _active_ws - 1)
        if vc_instance is not None:
            del vc_instance
            vc_instance = None

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            await ws.close()
        except Exception:
            pass
