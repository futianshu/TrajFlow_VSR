"""Top-level TrajFlow-VSR model skeleton."""

from __future__ import annotations

from typing import Any

from trajflow_vsr.models.consistency.data_projection import ReliabilityCalibratedDataConsistency
from trajflow_vsr.models.decoder.wavelet_operator_decoder import SpacetimeWaveletOperatorDecoder
from trajflow_vsr.models.flow.rectified_flow import ConditionalRectifiedFlowResidualGenerator
from trajflow_vsr.models.memory.trajectory_koopman_ssm import TrajectoryKoopmanSSMMemory
from trajflow_vsr.models.tokenizer.evidence_tokenizer import MultiScaleEvidenceTokenizer
from trajflow_vsr.models.transport.ot_sb_bridge import OTSBTrajectoryBridge
from trajflow_vsr.models.uncertainty.degradation_encoder import DegradationCausalUncertaintyEncoder
from trajflow_vsr.utils.torch_utils import require_torch_nn


class TrajFlowVSR:
    """Reliability-calibrated conditional transport model for VSR."""

    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = require_torch_nn()

        class _TrajFlowVSR(nn.Module):
            def __init__(
                self,
                hidden_channels: int = 64,
                in_channels: int = 3,
                out_channels: int = 3,
                transport_temperature: float = 0.2,
                transport_sinkhorn_iterations: int = 12,
                transport_unbalanced_floor: float = 0.05,
                transport_bridge_steps: int = 3,
                transport_spatial_radius: int = 4,
                transport_temporal_radius: int = 2,
                transport_use_unbalanced: bool = True,
                transport_use_reliability: bool = True,
                memory_scan_policy: str = "ot_sb",
                memory_use_koopman: bool = True,
                memory_trajectory_topk: int = 4,
                flow_enabled: bool = False,
                flow_gate_max: float = 0.25,
                flow_reliability_weight: float = 1.0,
                flow_texture_uncertainty_weight: float = 1.0,
                flow_amplitude_limit: float = 0.25,
                flow_bandlimit_kernel_size: int = 3,
                decoder_use_operator: bool = True,
                decoder_use_fourier: bool = True,
                decoder_use_coordinate: bool = True,
                decoder_use_wavelet: bool = True,
                decoder_use_anti_aliasing: bool = True,
                consistency_strength: float = 0.2,
            ):
                super().__init__()
                self.flow_enabled = bool(flow_enabled)
                self.tokenizer = MultiScaleEvidenceTokenizer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                )
                self.uncertainty = DegradationCausalUncertaintyEncoder(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                )
                self.transport = OTSBTrajectoryBridge(
                    hidden_channels=hidden_channels,
                    temperature=transport_temperature,
                    sinkhorn_iterations=transport_sinkhorn_iterations,
                    unbalanced_floor=transport_unbalanced_floor,
                    bridge_steps=transport_bridge_steps,
                    spatial_radius=transport_spatial_radius,
                    temporal_radius=transport_temporal_radius,
                    use_unbalanced=transport_use_unbalanced,
                    use_reliability=transport_use_reliability,
                )
                self.memory = TrajectoryKoopmanSSMMemory(
                    hidden_channels=hidden_channels,
                    scan_policy=memory_scan_policy,
                    use_koopman=memory_use_koopman,
                    trajectory_topk=memory_trajectory_topk,
                )
                self.flow = ConditionalRectifiedFlowResidualGenerator(
                    hidden_channels=hidden_channels,
                    gate_max=flow_gate_max,
                    reliability_weight=flow_reliability_weight,
                    texture_uncertainty_weight=flow_texture_uncertainty_weight,
                    amplitude_limit=flow_amplitude_limit,
                    bandlimit_kernel_size=flow_bandlimit_kernel_size,
                )
                self.decoder = SpacetimeWaveletOperatorDecoder(
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    use_operator=decoder_use_operator,
                    use_fourier=decoder_use_fourier,
                    use_coordinate=decoder_use_coordinate,
                    use_wavelet=decoder_use_wavelet,
                    use_anti_aliasing=decoder_use_anti_aliasing,
                )
                self.consistency = ReliabilityCalibratedDataConsistency(
                    strength=consistency_strength,
                )

            def forward(
                self,
                lr_video,
                scale: float = 4.0,
                mode: str = "offline",
                sample_flow_noise: bool = False,
                distill_flow: bool = False,
                flow_teacher_steps: int = 4,
            ) -> dict[str, Any]:
                if mode not in {"offline", "streaming"}:
                    raise ValueError(f"Unsupported inference mode: {mode}")

                token_bundle = self.tokenizer(lr_video, scale=scale)
                uncertainty = self.uncertainty(lr_video)
                transport = self.transport(token_bundle, uncertainty, causal=mode == "streaming")
                memory = self.memory(transport, uncertainty)
                if self.flow_enabled:
                    residual = self.flow(
                        memory,
                        uncertainty,
                        sample_noise=sample_flow_noise,
                        distill=distill_flow,
                        teacher_steps=flow_teacher_steps,
                    )
                else:
                    residual = {
                        "residual_grid": memory["memory_grid"].new_zeros(memory["memory_grid"].shape),
                        "flow_disabled": True,
                    }
                decoded = self.decoder(lr_video, memory, residual, scale=scale)
                projected = self.consistency(lr_video, decoded, uncertainty, scale=scale)
                return {
                    **projected,
                    "mode": mode,
                    "uncertainty": uncertainty,
                    "transport": transport,
                    "memory": memory,
                    "residual": residual,
                    "decoded": decoded,
                }

        return _TrajFlowVSR(*args, **kwargs)


def build_trajflow_vsr(config: dict[str, Any]) -> Any:
    """Build the top-level model from config."""

    tokenizer_cfg = config.get("tokenizer", {})
    transport_cfg = config.get("transport", {})
    memory_cfg = config.get("memory", {})
    flow_cfg = config.get("flow", {})
    decoder_cfg = config.get("decoder", {})
    consistency_cfg = config.get("consistency", {})
    return TrajFlowVSR(
        hidden_channels=int(config.get("hidden_channels", tokenizer_cfg.get("hidden_channels", 64))),
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 3)),
        transport_temperature=float(transport_cfg.get("temperature", 0.2)),
        transport_sinkhorn_iterations=int(transport_cfg.get("sinkhorn_iterations", 12)),
        transport_unbalanced_floor=float(transport_cfg.get("unbalanced_floor", 0.05)),
        transport_bridge_steps=int(transport_cfg.get("bridge_steps", 3)),
        transport_spatial_radius=int(transport_cfg.get("spatial_radius", 4)),
        transport_temporal_radius=int(transport_cfg.get("temporal_radius", 2)),
        transport_use_unbalanced=bool(transport_cfg.get("use_unbalanced", True)),
        transport_use_reliability=bool(transport_cfg.get("use_reliability", True)),
        memory_scan_policy=str(memory_cfg.get("scan_policy", "ot_sb")),
        memory_use_koopman=bool(memory_cfg.get("use_koopman", True)),
        memory_trajectory_topk=int(memory_cfg.get("trajectory_topk", 4)),
        flow_enabled=bool(flow_cfg.get("enabled", False)),
        flow_gate_max=float(flow_cfg.get("gate_max", 0.25)),
        flow_reliability_weight=float(flow_cfg.get("reliability_weight", 1.0)),
        flow_texture_uncertainty_weight=float(flow_cfg.get("texture_uncertainty_weight", 1.0)),
        flow_amplitude_limit=float(flow_cfg.get("amplitude_limit", 0.25)),
        flow_bandlimit_kernel_size=int(flow_cfg.get("bandlimit_kernel_size", 3)),
        decoder_use_operator=bool(decoder_cfg.get("use_operator", True)),
        decoder_use_fourier=bool(decoder_cfg.get("use_fourier", True)),
        decoder_use_coordinate=bool(decoder_cfg.get("use_coordinate", True)),
        decoder_use_wavelet=bool(decoder_cfg.get("use_wavelet", True)),
        decoder_use_anti_aliasing=bool(decoder_cfg.get("use_anti_aliasing", True)),
        consistency_strength=float(consistency_cfg.get("strength", 0.2)),
    )
