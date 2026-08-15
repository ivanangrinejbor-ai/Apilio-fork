"""
Convert the official charactr Vocos 24 kHz mel checkpoint into an RVC
pretrainG checkpoint.

The project's VocosDecoder (rvc/lib/algorithm/generators/vocos_dec.py) is an
architectural port of the official Vocos model (ConvNeXt backbone + ISTFT
head) with the same dimensions (dim 512, intermediate 1536, 8 layers), so the
official weights transfer 1:1 except:

  - backbone.embed (input projection): official projects 100 mel bands, the
    RVC decoder projects the 192-channel latent -> kept randomly initialized
    and trained from scratch
  - feature_extractor (mel filterbank): not part of the RVC decoder

The decoder is built with upsample_rates [4,4,2,2,2,2] so its hop length
(256) matches the official model, and the RVC inference window is derived
from the checkpoint config (171 at 24 kHz, as for BigVGAN-v2).

Usage:
    python rvc/lib/tools/convert_vocos.py \
        --output assets/pretrained_vocos/pretrained_vocos_24k.pth

    (without --input it downloads charactr/vocos-mel-24khz from Hugging Face)
"""

import argparse
import os
import sys

import torch

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402

HF_REPO = "charactr/vocos-mel-24khz"
HF_FILENAME = "pytorch_model.bin"

SR = 24000
HOP_LENGTH = 256
SEGMENT_SIZE = 8192
FILTER_LENGTH = 1024
SPK_EMBED_DIM = 109

# Official keys that have no counterpart in the RVC decoder.
SKIP_PREFIXES = (
    "feature_extractor.",
    "backbone.embed.",
)

UPSAMPLE_RATES_24K = [4, 4, 2, 2, 2, 2]


def build_synthesizer(spk_embed_dim: int) -> Synthesizer:
    return Synthesizer(
        spec_channels=FILTER_LENGTH // 2 + 1,
        segment_size=SEGMENT_SIZE // HOP_LENGTH,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0,
        resblock="1",
        upsample_rates=UPSAMPLE_RATES_24K,
        upsample_initial_channel=0,
        upsample_kernel_sizes=[0],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        spk_embed_dim=spk_embed_dim,
        gin_channels=256,
        use_spectral_norm=False,
        use_f0=True,
        sr=SR,
        vocoder="Vocos",
    )


def convert(
    input_path: str, output_path: str, spk_embed_dim: int = SPK_EMBED_DIM
) -> None:
    if not os.path.isfile(input_path):
        sys.exit(f"Checkpoint not found: {input_path}")

    print(f"Loading official Vocos checkpoint: {input_path}")
    official_state = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(official_state, dict):
        sys.exit("Unrecognized checkpoint format: expected a plain state dict")

    print("Building RVC Synthesizer (24 kHz, Vocos decoder)...")
    net_g = build_synthesizer(spk_embed_dim)

    dec_state = net_g.dec.state_dict()
    mapped = {}
    skipped = []
    for key, value in official_state.items():
        if key.startswith(SKIP_PREFIXES):
            skipped.append(key)
            continue
        target = key
        if target not in dec_state:
            sys.exit(f"Key mismatch: '{key}' -> '{target}' not in project decoder")
        if tuple(value.shape) != tuple(dec_state[target].shape):
            sys.exit(
                f"Shape mismatch: '{key}' -> '{target}' "
                f"({tuple(value.shape)} vs {tuple(dec_state[target].shape)})"
            )
        mapped[target] = value

    net_g.dec.load_state_dict(mapped, strict=False)

    # Warm-start the input projection (backbone.embed): the official Vocos
    # projects 100 mel bands, the RVC decoder projects the 192-channel latent.
    # Both are Conv1d with kernel 7, so the official weights slot into the
    # first 100 input channels and the pretrained backbone receives a
    # mel-like projection from the first step instead of pure noise.
    try:
        official_w = official_state["backbone.embed.weight"]
        official_b = official_state["backbone.embed.bias"]
        if tuple(official_w.shape) == (512, 100, 7):
            with torch.no_grad():
                proj_w = net_g.dec.backbone.embed.weight
                proj_b = net_g.dec.backbone.embed.bias
                proj_w.zero_()
                proj_w[:, :100, :] = official_w
                proj_b.copy_(official_b)
            print(
                "Warm-started backbone.embed: official mel projection copied "
                "into the first 100 latent channels."
            )
    except KeyError:
        print("NOTE: official backbone.embed not found; input projection stays random.")

    missing = sorted(set(dec_state) - set(mapped))
    print(
        f"Transferred {len(mapped)} tensors "
        f"({len(skipped)} skipped: {', '.join(skipped)})."
    )
    if missing:
        print(
            f"Left randomly initialized ({len(missing)}): "
            + ", ".join(missing[:8])
            + ("..." if len(missing) > 8 else "")
        )

    full = net_g.state_dict()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(
        {
            "model": full,
            "epoch": 0,
            "iteration": 0,
            "optimizer": {},
            "scaler": {},
            "learning_rate": 1e-4,
        },
        output_path,
    )
    print(f"Saved pretrainG checkpoint: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert the official charactr Vocos 24 kHz checkpoint into an RVC pretrainG checkpoint."
    )
    parser.add_argument("--input", type=str, help="Path to pytorch_model.bin")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=HF_REPO,
        help="Hugging Face repo id to download the checkpoint from (used when --input is not given).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/pretrained_vocos/pretrained_vocos_24k.pth",
        help="Output RVC pretrainG checkpoint path.",
    )
    parser.add_argument(
        "--spk-embed-dim",
        type=int,
        default=SPK_EMBED_DIM,
        help="Speaker embedding size (default: 109).",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            sys.exit(f"huggingface_hub is required for --hf-repo: {error}")
        input_path = hf_hub_download(args.hf_repo, HF_FILENAME)
        print(f"Downloaded: {input_path}")

    convert(input_path, args.output, args.spk_embed_dim)


if __name__ == "__main__":
    main()
