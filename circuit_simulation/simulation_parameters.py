"""Memory-experiment conditions, decoder parameters, and sampling tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Basis = Literal["Z", "X"]
DecodingMode = Literal["xyz", "xz"]


def format_p(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class RelayBPConfig:
    gamma0: float = 0.1
    pre_iter: int = 200
    num_sets: int = 10
    set_max_iter: int = 200
    gamma_dist_min: float = -0.24
    gamma_dist_max: float = 0.6
    stop_nconv: int = 5

    def kwargs(self) -> dict[str, object]:
        return {
            "gamma0": float(self.gamma0),
            "pre_iter": int(self.pre_iter),
            "num_sets": int(self.num_sets),
            "set_max_iter": int(self.set_max_iter),
            "gamma_dist_interval": (
                float(self.gamma_dist_min),
                float(self.gamma_dist_max),
            ),
            "stop_nconv": int(self.stop_nconv),
        }


@dataclass(frozen=True)
class RelayBPFallbackConfig:
    enabled: bool = False
    gamma0: float = 0.2
    pre_iter: int = 300
    num_sets: int = 50
    set_max_iter: int = 150
    gamma_dist_min: float = -0.24
    gamma_dist_max: float = 0.6
    stop_nconv: int = 4

    def relay_config(self) -> RelayBPConfig:
        return RelayBPConfig(
            gamma0=float(self.gamma0),
            pre_iter=int(self.pre_iter),
            num_sets=int(self.num_sets),
            set_max_iter=int(self.set_max_iter),
            gamma_dist_min=float(self.gamma_dist_min),
            gamma_dist_max=float(self.gamma_dist_max),
            stop_nconv=int(self.stop_nconv),
        )


@dataclass(frozen=True)
class BpOsdRetryConfig:
    enabled: bool = False
    max_iter: int = 300
    bp_method: str = "minimum_sum"
    ms_scaling_factor: float = 0.0
    schedule: str = "serial"
    osd_method: str = "OSD_CS"
    osd_order: int = 1
    threads: int = 8

    def kwargs(self) -> dict[str, object]:
        return {
            "max_iter": int(self.max_iter),
            "bp_method": str(self.bp_method),
            "ms_scaling_factor": float(self.ms_scaling_factor),
            "schedule": str(self.schedule),
            "osd_method": str(self.osd_method),
            "osd_order": int(self.osd_order),
            "input_vector_type": "syndrome",
            "omp_thread_count": int(self.threads),
        }


@dataclass(frozen=True)
class MIPFallbackConfig:
    enabled: bool = False
    time_limit: float = 0.0
    wall_time_limit: float = 0.0
    mip_rel_gap: float = 0.0
    feasible_only: bool = False
    count_unsolved_as_failure: bool = True
    workers: int = 1
    threads: int = 1

    def options(self) -> dict[str, object]:
        options: dict[str, object] = {}
        if self.time_limit > 0:
            options["time_limit"] = float(self.time_limit)
        if self.mip_rel_gap > 0:
            options["mip_rel_gap"] = float(self.mip_rel_gap)
        if self.threads > 0:
            options["threads"] = int(self.threads)
        return options


@dataclass(frozen=True)
class Condition:
    code_name: str
    parameter_label: str
    p_code: int
    l: int
    j: int
    expected_d: int
    basis: Basis
    decoding_mode: DecodingMode
    p_noise: float
    cycles: int

    @property
    def label(self) -> str:
        return f"{self.code_name}_{self.basis.lower()}basis_{self.decoding_mode}_p{format_p(self.p_noise)}_c{self.cycles}"


@dataclass(frozen=True)
class GenerateTask:
    chunk_index: int
    start_shot: int
    num_shots: int
    seed: int
    output_path: str
    num_detectors: int
    num_observables: int


@dataclass(frozen=True)
class DecodeTask:
    chunk_index: int
    shot_path: str


DecoderName = Literal["relaybp"]
SCRIPT_VERSION = 1
DECODE_SCHEMA_VERSION = 8
