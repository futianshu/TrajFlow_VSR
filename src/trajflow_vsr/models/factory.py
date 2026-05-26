"""Model factory and lightweight summaries."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from trajflow_vsr.models.common import component_spec
from trajflow_vsr.models.tokenizer.stage_a_pretrainer import build_stage_a_tokenizer_pretrainer
from trajflow_vsr.models.trajflow_vsr import build_trajflow_vsr


def describe_model(config: dict[str, Any]) -> dict[str, Any]:
    """Return a model summary without instantiating heavy dependencies."""

    hidden = int(config.get("hidden_channels", 64))
    components = [
        component_spec(
            "MultiScaleEvidenceTokenizer",
            "LR video -> structure/texture/artifact evidence tokens",
            config.get("tokenizer", {"hidden_channels": hidden}),
        ),
        component_spec(
            "DegradationCausalUncertaintyEncoder",
            "artifact/reliability/motion/texture uncertainty estimation",
            config.get("uncertainty", {"hidden_channels": hidden}),
        ),
        component_spec(
            "OTSBTrajectoryBridge",
            "reliability-calibrated soft trajectory transport",
            config.get("transport", {}),
        ),
        component_spec(
            "TrajectoryKoopmanSSMMemory",
            "long-context trajectory memory and Koopman regularization",
            config.get("memory", {"hidden_channels": hidden}),
        ),
        component_spec(
            "ConditionalRectifiedFlowResidualGenerator",
            "optional reliability-gated posterior transport for HR high-frequency residuals",
            config.get("flow", {"hidden_channels": hidden}),
        ),
        component_spec(
            "SpacetimeWaveletOperatorDecoder",
            "arbitrary-scale operator decoding with anti-aliasing",
            config.get("decoder", {"hidden_channels": hidden}),
        ),
        component_spec(
            "ReliabilityCalibratedDataConsistency",
            "low-frequency evidence projection controlled by reliability",
            config.get("consistency", {}),
        ),
    ]
    return {
        "name": config.get("name", "trajflow_vsr"),
        "hidden_channels": hidden,
        "components": [asdict(component) for component in components],
        "pretrained": _pretrained_config_summary(config),
    }


def build_model(config: dict[str, Any]) -> Any:
    """Build a model by name."""

    name = config.get("name", "trajflow_vsr")
    if name != "trajflow_vsr":
        raise ValueError(f"Unknown model name: {name}")
    return build_trajflow_vsr(config)


def build_stage_model(stage: dict[str, Any], config: dict[str, Any]) -> Any:
    """Build the model used by a training stage."""

    if stage.get("name") == "stage_a_tokenizer":
        return build_stage_a_tokenizer_pretrainer(config)
    return build_model(config)


def _pretrained_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    pretrained = config.get("pretrained", {})
    if not isinstance(pretrained, dict) or not pretrained.get("path"):
        return {"enabled": False}
    components = pretrained.get("components", ["tokenizer", "uncertainty"])
    freeze_components = pretrained.get("freeze_components", [])
    if isinstance(components, str):
        components = [components]
    if isinstance(freeze_components, str):
        freeze_components = [freeze_components]
    return {
        "enabled": True,
        "path": pretrained.get("path"),
        "components": [str(item) for item in components],
        "freeze_components": [str(item) for item in freeze_components],
        "strict": bool(pretrained.get("strict", False)),
    }
