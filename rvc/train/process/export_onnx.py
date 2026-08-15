import os
import pickle
import sys

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.utils import validate_ui_path, remap_weight_norm_keys


class _InferWrapper(torch.nn.Module):
    def __init__(self, net_g, use_f0):
        super().__init__()
        self.net_g = net_g
        self.use_f0 = use_f0

    def forward(self, phone, phone_lengths, pitch, nsff0, sid):
        if self.use_f0:
            return self.net_g.infer(phone, phone_lengths, pitch, nsff0, sid, None)
        return self.net_g.infer(phone, phone_lengths, None, None, sid, None)


def export_onnx(ckpt_path: str, output_dir: str | None = None, frames: int = 512) -> str:
    ckpt_path = validate_ui_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if frames < 1:
        raise ValueError("frames must be >= 1")

    try:
        cpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to load checkpoint '{ckpt_path}': {e}")
    if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt:
        raise ValueError(f"'{ckpt_path}' is not a valid RVC checkpoint")

    sr = cpt.get("sr", cpt["config"][-1])
    use_f0 = bool(cpt.get("f0", 1))
    version = cpt.get("version", "v2")
    text_enc_hidden_dim = 768 if version == "v2" else 256
    vocoder = cpt.get("vocoder", "HiFi-GAN")

    net_g = Synthesizer(
        *cpt["config"],
        use_f0=use_f0,
        text_enc_hidden_dim=text_enc_hidden_dim,
        vocoder=vocoder,
    )
    del net_g.enc_q
    net_g.load_state_dict(
        remap_weight_norm_keys(cpt["weight"]), strict=False
    )
    net_g.eval()

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(ckpt_path))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(ckpt_path))[0] + ".onnx"
    )

    phone = torch.randn(1, frames, text_enc_hidden_dim)
    phone_lengths = torch.tensor([frames], dtype=torch.long)
    sid = torch.tensor([0], dtype=torch.long)
    pitch = torch.randint(0, 256, (1, frames), dtype=torch.long)
    nsff0 = torch.rand(1, frames)
    args = (phone, phone_lengths, pitch, nsff0, sid)

    from torch.export import Dim

    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = {
        "phone": {0: batch},
        "phone_lengths": {0: batch},
        "pitch": {0: batch},
        "nsff0": {0: batch},
        "sid": {0: batch},
    }

    with torch.no_grad():
        torch.onnx.export(
            _InferWrapper(net_g, use_f0),
            args,
            output_path,
            input_names=["phone", "phone_lengths", "pitch", "nsff0", "sid"],
            output_names=["audio", "x_mask"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
        )

    hop = 1
    for r in cpt["config"][12]:
        hop *= r
    audio_len = frames * hop
    print(
        f"Exported ONNX model to '{output_path}' "
        f"(sr={sr}, f0={use_f0}, vocoder={vocoder}, "
        f"fixed length={frames} frames = {audio_len / sr:.2f}s of audio)"
    )
    return output_path