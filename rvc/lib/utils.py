import os
import sys
import socket
import tempfile
import urllib.parse
import soxr
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
from torch import nn

import logging
from transformers import HubertModel
import warnings

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


def remap_weight_norm_keys(state_dict):
    """
    Convert legacy weight_norm keys (weight_g/weight_v) to the current
    parametrization keys (.parametrizations.weight.original0/original1).

    RVC-ecosystem checkpoints (official pretrains, extracted trained models,
    blender outputs) store the legacy keys; the Synthesizer builds weight-norm
    layers with torch parametrizations. Loading legacy keys without remapping
    silently drops every weight-norm tensor (strict=False), leaving those
    layers random and producing garbage audio.
    """
    return {
        k.replace(".weight_v", ".parametrizations.weight.original1").replace(
            ".weight_g", ".parametrizations.weight.original0"
        ): v
        for k, v in state_dict.items()
    }


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_audio_16k(file):
    # this is used by f0 and feature extractions that load preprocessed 16k files, so there's no need to resample
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file,
    sample_rate,
    **kwargs,
):
    formant_shifting = kwargs.get("formant_shifting", False)
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
        if formant_shifting:
            formant_qfrency = kwargs.get("formant_qfrency", 0.8)
            formant_timbre = kwargs.get("formant_timbre", 0.8)

            from stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(
                audio,
                factors=1,
                quefrency=formant_qfrency * 1e-3,
                distortion=formant_timbre,
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


def format_title(title):
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title, flags=re.UNICODE)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def safe_extract_zip(zip_ref, dest_path):
    """Extract a zip archive while blocking zip-slip (path traversal) members."""
    dest_real = os.path.realpath(dest_path)
    for member in zip_ref.infolist():
        member_path = os.path.realpath(os.path.join(dest_path, member.filename))
        if not member_path.startswith(dest_real + os.sep):
            raise ValueError(f"Unsafe path in zip archive: {member.filename}")
    zip_ref.extractall(dest_path)


def ensure_within_root(path, root):
    """Return the absolute path if it stays inside root, raise ValueError otherwise."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root)
    if abs_path != abs_root and not abs_path.startswith(abs_root + os.sep):
        raise ValueError(f"Path {path} is outside the allowed directory {root}")
    return abs_path


def validate_ui_path(path, root=None):
    """Validate a user-supplied path: must be inside the project root,
    or a file uploaded through Gradio (stored in its temp directory)."""
    if not path:
        return path
    abs_path = os.path.abspath(str(path))
    root = os.path.abspath(root or now_dir)
    gradio_dir = os.path.abspath(os.path.join(tempfile.gettempdir(), "gradio"))
    if abs_path == gradio_dir or abs_path.startswith(gradio_dir + os.sep):
        return abs_path
    return ensure_within_root(abs_path, root)


def validate_url(url):
    """Reject SSRF-prone URLs (non-http(s), localhost, private/link-local addresses)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are allowed: {url}")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"Invalid URL: {url}")
    hostname = hostname.lower()
    if hostname in ("localhost",):
        raise ValueError(f"Local addresses are not allowed: {url}")
    try:
        ip = socket.gethostbyname(hostname)
    except OSError:
        raise ValueError(f"Could not resolve host: {hostname}")
    parts = [int(p) for p in ip.split(".")]
    if (
        parts[0] == 10
        or parts[0] == 127
        or (parts[0] == 169 and parts[1] == 254)
        or (parts[0] == 172 and 16 <= parts[1] <= 31)
        or (parts[0] == 192 and parts[1] == 168)
        or (parts[0] == 0)
        or parts[0] >= 224
    ):
        raise ValueError(f"Private or reserved addresses are not allowed: {url}")
    return url


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin": os.path.join(embedder_root, "spin"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/pytorch_model.bin",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/config.json",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/config.json",
    }

    if embedder_model == "custom":
        if os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    return models
