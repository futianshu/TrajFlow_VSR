"""Export OT/SB transport-plan diagnostics for a config/checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.data.factory import CONTROLLED_MOTION_NAME, SYNTHETIC_NAME, build_synthetic_spec  # noqa: E402
from trajflow_vsr.data.manifest import make_frame_manifest_batch  # noqa: E402
from trajflow_vsr.data.synthetic import make_controlled_motion_batch, make_synthetic_batch  # noqa: E402
from trajflow_vsr.diagnostics import transport_plan_diagnostics  # noqa: E402
from trajflow_vsr.models.factory import build_model  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402
from trajflow_vsr.utils.seed import seed_everything  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training/evaluation config used to build data and model.")
    parser.add_argument("--checkpoint", help="Optional model checkpoint to load.")
    parser.add_argument("--output-dir", default="outputs/diagnostics/transport", help="Directory for diagnostics JSON.")
    parser.add_argument("--name", default="transport_diagnostics", help="Output file stem.")
    parser.add_argument("--mode", default="offline", choices=["offline", "streaming"], help="Forward mode.")
    parser.add_argument("--device", help="Override runtime.device.")
    parser.add_argument("--seed", type=int, help="Override runtime.seed.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Dotted config override, e.g. data.name=controlled_motion. Can repeat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    runtime = config.setdefault("runtime", {})
    if args.device:
        runtime["device"] = args.device
    if args.seed is not None:
        runtime["seed"] = int(args.seed)

    torch = require_torch()
    seed_everything(int(runtime.get("seed", 20260524)))
    device = str(runtime.get("device", "cpu"))
    model = build_model(config.get("model", {})).to(device)
    checkpoint = _load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    data_config = config.get("data", {})
    batch = _make_batch(data_config, device=device)
    with torch.no_grad():
        outputs = model(batch["lr"], scale=batch["scale"], mode=args.mode)

    diagnostics = transport_plan_diagnostics(
        outputs["transport"],
        frames=int(batch["lr"].shape[1]),
        height=int(batch["lr"].shape[-2]),
        width=int(batch["lr"].shape[-1]),
        shift_x=int(batch.get("controlled_motion", {}).get("shift_x", data_config.get("motion", {}).get("shift_x", 0))),
        shift_y=int(batch.get("controlled_motion", {}).get("shift_y", data_config.get("motion", {}).get("shift_y", 0))),
        reliability=outputs.get("uncertainty", {}).get("reliability"),
    )
    result = {
        "config": args.config,
        "checkpoint": checkpoint,
        "mode": args.mode,
        "data": {
            "name": data_config.get("name", SYNTHETIC_NAME),
            "frames": int(batch["lr"].shape[1]),
            "height": int(batch["lr"].shape[-2]),
            "width": int(batch["lr"].shape[-1]),
            "scale": float(batch["scale"]),
            "controlled_motion": batch.get("controlled_motion"),
            "metadata": batch.get("metadata", []),
        },
        "diagnostics": diagnostics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.name}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output_path"] = str(output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _make_batch(data_config: dict, device: str):
    name = data_config.get("name", SYNTHETIC_NAME)
    if name == "frame_manifest":
        return make_frame_manifest_batch(data_config, device=device)
    spec = build_synthetic_spec(data_config)
    if name == CONTROLLED_MOTION_NAME:
        return make_controlled_motion_batch(spec, motion=data_config.get("motion", {}), device=device)
    return make_synthetic_batch(spec, device=device)


def _load_checkpoint(model, checkpoint_path: str | None, device: str) -> dict:
    if not checkpoint_path:
        return {"loaded": False, "path": None}
    torch = require_torch()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


if __name__ == "__main__":
    main()
