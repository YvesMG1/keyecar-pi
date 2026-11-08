"""Export a training checkpoint as an ONNX bundle for the car.

    python -m keyecar.training.export runs/rgb_history_dagger_v4/best.pt --out-dir models/rgb_history_dagger_v4

The bundle contains::

    policy.onnx             # FP32, opset 17
    policy.int8.onnx        # dynamic INT8 weight quantization (unless --no-int8)
    policy_metadata.json    # variant, preprocessing, and training provenance

ONNX inputs: ``image`` (N, 3 * history, H, W), ``motor_state`` (N, 2),
``condition`` (N,), and for history models ``condition_history`` (N, history).
The output is (N, 2): throttle and steering. After export the ONNX model is
checked against PyTorch on a random input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from keyecar.cli import add_checkpoint, make_parser

from .checkpoint import load_checkpoint, model_from_checkpoint
from .files import write_json
from .model import RegressionPolicy


def example_inputs(model: RegressionPolicy, image_size: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Random model inputs keyed by their ONNX input names, in the model's argument order."""
    rng = np.random.default_rng(seed)
    inputs = {
        "image": rng.standard_normal((1, model.input_channels, image_size, image_size)).astype(np.float32),
        "motor_state": rng.standard_normal((1, 2)).astype(np.float32),
        "condition": np.zeros(1, dtype=np.int64),
    }
    if model.condition_history_length > 0:
        inputs["condition_history"] = np.zeros((1, model.condition_history_length), dtype=np.int64)
    return inputs


def export_onnx(model: RegressionPolicy, output_path: Path, *, image_size: int) -> None:
    model.eval()
    inputs = example_inputs(model, image_size)
    torch.onnx.export(
        model,
        tuple(torch.from_numpy(value) for value in inputs.values()),
        str(output_path),
        input_names=list(inputs),
        output_names=["output"],
        dynamic_axes={name: {0: "batch"} for name in [*inputs, "output"]},
        opset_version=17,
        dynamo=False,  # the TorchScript exporter, which produced the shipped models
    )


def check_onnx_against_torch(model: RegressionPolicy, onnx_path: Path, *, image_size: int) -> float:
    """Return the largest output difference between ONNX Runtime and PyTorch."""
    inputs = example_inputs(model, image_size)
    with torch.no_grad():
        expected = model(*(torch.from_numpy(value) for value in inputs.values())).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)[0]
    return float(np.max(np.abs(expected - actual)))


def quantize(in_path: Path, out_path: Path) -> None:
    quantize_dynamic(model_input=str(in_path), model_output=str(out_path), weight_type=QuantType.QInt8)


def export_bundle(checkpoint_path: Path, out_dir: Path, *, int8: bool = True, check: bool = True) -> Path:
    """Write ``policy.onnx`` (and ``policy.int8.onnx``) plus ``policy_metadata.json`` to ``out_dir``."""
    state = load_checkpoint(checkpoint_path)
    metadata = dict(state["metadata"])
    image_size = int(metadata["image_size"])
    model = model_from_checkpoint(state)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "policy.onnx"
    export_onnx(model, onnx_path, image_size=image_size)
    print(f"Wrote {onnx_path}")
    if check:
        max_diff = check_onnx_against_torch(model, onnx_path, image_size=image_size)
        print(f"Largest difference between PyTorch and ONNX: {max_diff:.2e}")
        if max_diff > 1e-3:
            raise RuntimeError(f"ONNX export differs from PyTorch by {max_diff}")
    int8_path = out_dir / "policy.int8.onnx"
    if int8:
        quantize(onnx_path, int8_path)
        print(f"Wrote {int8_path}")

    metadata["export"] = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "int8_onnx": str(int8_path) if int8 else None,
    }
    metadata_path = out_dir / "policy_metadata.json"
    write_json(metadata_path, metadata)
    print(f"Wrote {metadata_path}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    add_checkpoint(parser)
    parser.add_argument("--out-dir", type=Path, required=True, help="bundle folder, e.g. models/<name>")
    parser.add_argument("--no-int8", action="store_true", help="skip INT8 quantization")
    parser.add_argument("--no-check", action="store_true", help="skip the ONNX vs. PyTorch comparison")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    export_bundle(args.checkpoint, args.out_dir, int8=not args.no_int8, check=not args.no_check)


if __name__ == "__main__":
    main()
