import os
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.infer import VoiceConverter
from assets.auth import check_api_token

app = FastAPI(title="Applio Infer API")
vc = VoiceConverter()


@app.middleware("http")
async def api_auth_middleware(request, call_next):
    from fastapi.responses import JSONResponse

    token = request.headers.get("Authorization", "")
    if token.startswith("Bearer "):
        token = token[len("Bearer ") :]
    if not check_api_token(token):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model_path: str = Form(...),
    index_path: str = Form(""),
    pitch: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    volume_envelope: float = Form(1.0),
    protect: float = Form(0.5),
    hop_length: int = Form(128),
    split_audio: bool = Form(False),
    split_audio_method: str = Form("threshold"),
    f0_autotune: bool = Form(False),
    f0_autotune_strength: float = Form(1.0),
    embedder_model: str = Form("contentvec"),
    clean_audio: bool = Form(False),
    clean_strength: float = Form(0.5),
    export_format: str = Form("WAV"),
    sid: int = Form(0),
):
    if not os.path.isfile(model_path):
        raise HTTPException(404, f"Model not found: {model_path}")

    suffix = os.path.splitext(file.filename or "input.wav")[1] or ".wav"
    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "input" + suffix)
        output_path = os.path.join(tmp, "output.wav")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        try:
            vc.convert_audio(
                audio_input_path=input_path,
                audio_output_path=output_path,
                model_path=model_path,
                index_path=index_path,
                pitch=pitch,
                f0_method=f0_method,
                index_rate=index_rate,
                volume_envelope=volume_envelope,
                protect=protect,
                hop_length=hop_length,
                split_audio=split_audio,
                split_audio_method=split_audio_method,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                embedder_model=embedder_model,
                clean_audio=clean_audio,
                clean_strength=clean_strength,
                export_format=export_format,
                sid=sid,
            )
        except Exception as error:
            raise HTTPException(500, f"Conversion failed: {error}")

        if not os.path.exists(output_path):
            raise HTTPException(500, "Conversion produced no output file.")

        with open(output_path, "rb") as f:
            audio_bytes = f.read()
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="output.wav"'},
    )
