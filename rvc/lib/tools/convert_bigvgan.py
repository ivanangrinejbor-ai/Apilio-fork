"""
Convert an official NVIDIA BigVGAN-v2 generator checkpoint into an Applio/RVC
pretrainG checkpoint ({"model": <Synthesizer state dict>}).

The official checkpoints (e.g. nvidia/bigvgan_v2_24khz_100band_256x) are
standalone mel->waveform generators using the old-style torch weight_norm
("weight_v"/"weight_g" keys), while this project's BigVGANGenerator is an NSF
variant using new-style parametrizations. This script remaps the shared
decoder blocks and leaves the NSF-specific modules (f0_upsample, m_source,
noise_convs, cond) and the RVC front-end (encoder/flow/emb_g) randomly
initialized for fine-tuning.

Usage:
    python rvc/lib/tools/convert_bigvgan.py \
        --input  /path/to/bigvgan_generator.pt \
        --output assets/pretrained_bigvgan/pretrained_bigvgan_24k.pth

    # or download straight from Hugging Face:
    python rvc/lib/tools/convert_bigvgan.py \
        --hf-repo nvidia/bigvgan_v2_24khz_100band_256x \
        --output assets/pretrained_bigvgan/pretrained_bigvgan_24k.pth
"""

import argparse
import os
import sys

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.algorithm.synthesizers import Synthesizer

# Architecture of nvidia/bigvgan_v2_24khz_100band_256x (from its config.json)
BIGVGAN_V2_24K = {
    "upsample_rates": [4, 4, 2, 2, 2, 2],
    "upsample_initial_channel": 1536,
    "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}

# RVC side: spec_channels / segment frames for the 24k pipeline (hop 256).
FILTER_LENGTH = 1024
SEGMENT_SIZE = 8192
HOP_LENGTH = 256


def _rename_antialias(rest: str) -> str:
    # official: activation_post.upsample / .downsample.lowpass
    # project:  act_post.up / act_post.down.lowpass
    rest = rest.replace("upsample.", "up.")
    rest = rest.replace("downsample.lowpass.", "down.lowpass.")
    return rest


def official_to_project(official_key: str) -> str | None:
    """
    Map an official BigVGAN-v2 state-dict key onto this project's
    BigVGANGenerator namespace, converting old-style weight_norm names.

    Returns None for modules that must stay randomly initialized
    (conv_pre input channels differ, NSF parts do not exist upstream).
    """
    key = official_key

    # conv_pre: official has num_mels (100) input channels, the RVC decoder
    # receives z (inter_channels=192) -> cannot transfer, keep random.
    if key.startswith("conv_pre."):
        return None

    if key.startswith("ups."):
        # official: ups.<i>.0.<param>  (ConvTranspose1d wrapped in Sequential)
        parts = key.split(".")
        assert parts[2] == "0", f"unexpected ups key: {key}"
        key = f"upsamples.{parts[1]}.{parts[3]}"
    elif key.startswith("resblocks."):
        # official: resblocks.<n>.<convs1|convs2|activations>.<j>.<param>
        # project:  amps.<stage>.<kernel>.layers.<layer>.<conv1|conv2|act1|act2>.<param>
        parts = key.split(".")
        stage, kernel = divmod(int(parts[1]), 3)
        if parts[2] == "convs1":
            key = f"amps.{stage}.{kernel}.layers.{parts[3]}.conv1.{parts[4]}"
        elif parts[2] == "convs2":
            key = f"amps.{stage}.{kernel}.layers.{parts[3]}.conv2.{parts[4]}"
        elif parts[2] == "activations":
            act = int(parts[3])
            layer, slot = divmod(act, 2)  # act 2i -> act1, act 2i+1 -> act2
            rest = _rename_antialias(".".join(parts[4:]))
            key = f"amps.{stage}.{kernel}.layers.{layer}.act{slot + 1}.{rest}"
        else:
            raise AssertionError(f"unexpected resblocks key: {key}")
    elif key.startswith("activation_post."):
        key = "act_post." + _rename_antialias(key[len("activation_post.") :])
    elif key.startswith("conv_post."):
        key = "conv_post." + key[len("conv_post.") :]
    else:
        raise AssertionError(f"unexpected official key: {key}")

    # Old-style weight_norm -> new-style parametrizations.
    # Old style: weight = g * v / ||v||  (keys: weight_v = v, weight_g = g).
    # New style (torch parametrizations.weight_norm): original0 = g, original1 = v,
    # which is why the project's load_checkpoint uses this same pairing.
    if key.endswith("weight_v"):
        key = key[: -len("weight_v")] + "parametrizations.weight.original1"
    elif key.endswith("weight_g"):
        key = key[: -len("weight_g")] + "parametrizations.weight.original0"

    return key


def build_synthesizer(spk_embed_dim: int) -> Synthesizer:
    config = dict(BIGVGAN_V2_24K)
    config.update(
        {
            "spec_channels": FILTER_LENGTH // 2 + 1,
            "segment_size": SEGMENT_SIZE // HOP_LENGTH,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "spk_embed_dim": spk_embed_dim,
            "gin_channels": 256,
            "use_spectral_norm": False,
            "use_f0": True,
            "sr": 24000,
            "vocoder": "BigVGAN",
        }
    )
    return Synthesizer(**config)


def convert(input_path: str, output_path: str, spk_embed_dim: int = 109) -> None:
    if not os.path.isfile(input_path):
        sys.exit(f"Checkpoint not found: {input_path}")

    print(f"Loading official BigVGAN checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)
    official_state = ckpt.get("generator", ckpt)
    if not isinstance(official_state, dict):
        sys.exit("Unrecognized checkpoint format: expected {'generator': {...}}")

    print("Building RVC Synthesizer (24 kHz, BigVGAN-v2 decoder)...")
    net_g = build_synthesizer(spk_embed_dim)

    dec_state = net_g.dec.state_dict()
    mapped = {}
    skipped = []
    for key, value in official_state.items():
        target = official_to_project(key)
        if target is None:
            skipped.append(key)
            continue
        if target not in dec_state:
            sys.exit(f"Key mismatch: '{key}' -> '{target}' not in project decoder")
        if tuple(value.shape) != tuple(dec_state[target].shape):
            # SnakeBeta parameters are stored flat (C,) upstream but as (1, C, 1)
            # in this project; same values, same broadcasting semantics.
            if (
                target.endswith((".act.alpha", ".act.beta"))
                and value.numel() == dec_state[target].numel()
            ):
                value = value.reshape(dec_state[target].shape)
            else:
                sys.exit(
                    f"Shape mismatch: '{key}' -> '{target}' "
                    f"({tuple(value.shape)} vs {tuple(dec_state[target].shape)})"
                )
        mapped[target] = value

    net_g.dec.load_state_dict(mapped, strict=False)

    # Warm-start conv_pre (input projection): the official BigVGAN maps 100
    # mel bands, the RVC decoder maps the 192-channel latent. Both are
    # Conv1d with kernel 7, so the official weight_norm parameters slot into
    # the first 100 input channels and the pretrained decoder blocks receive
    # a mel-like projection from the first step instead of pure noise
    # (zeroed channels contribute nothing to the weight_norm ||v||, so the
    # official effective weights are preserved exactly).
    try:
        official_w = official_state["conv_pre.weight_v"]
        official_g = official_state["conv_pre.weight_g"]
        official_b = official_state["conv_pre.bias"]
        if tuple(official_w.shape) == (1536, 100, 7):
            with torch.no_grad():
                proj_v = net_g.dec.conv_pre.parametrizations.weight.original1
                proj_g = net_g.dec.conv_pre.parametrizations.weight.original0
                proj_b = net_g.dec.conv_pre.bias
                proj_v.zero_()
                proj_v[:, :100, :] = official_w
                proj_g.copy_(official_g)
                proj_b.copy_(official_b)
            print(
                "Warm-started conv_pre: official mel projection copied "
                "into the first 100 latent channels."
            )
    except KeyError:
        print("NOTE: official conv_pre not found; input projection stays random.")

    missing = sorted(set(dec_state) - set(mapped))
    unexpected = sorted(set(mapped) - set(dec_state))
    print(f"Transferred {len(mapped)} tensors ({len(skipped)} skipped: conv_pre).")
    if missing:
        print(
            f"Left randomly initialized ({len(missing)}): "
            + ", ".join(missing[:8])
            + ("..." if len(missing) > 8 else "")
        )
    if unexpected:
        sys.exit(f"Unexpected mapped keys: {unexpected}")

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
        description="Convert an NVIDIA BigVGAN-v2 24 kHz generator into an RVC pretrainG checkpoint."
    )
    parser.add_argument("--input", type=str, help="Path to bigvgan_generator.pt")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="nvidia/bigvgan_v2_24khz_100band_256x",
        help="Hugging Face repo id to download the checkpoint from (used when --input is not given).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/pretrained_bigvgan/pretrained_bigvgan_24k.pth",
        help="Output RVC pretrainG checkpoint path.",
    )
    parser.add_argument(
        "--spk-embed-dim", type=int, default=109, help="Speaker embedding size (default: 109)."
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            sys.exit(f"huggingface_hub is required for --hf-repo: {error}")
        input_path = hf_hub_download(args.hf_repo, "bigvgan_generator.pt")
        print(f"Downloaded: {input_path}")

    convert(input_path, args.output, args.spk_embed_dim)


if __name__ == "__main__":
    main()
