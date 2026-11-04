"""Pydantic Schemas for Physical Configuration, K-Space State, and WebSocket Telemetry."""

from __future__ import annotations
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class PhantomConfigRequest(BaseModel):
    """Parametros para generar o actualizar el espacio-k y la mascara de aceleracion."""

    height: int = Field(default=128, ge=64, le=256)
    width: int = Field(default=128, ge=64, le=256)
    num_coils: int = Field(default=8, ge=2, le=32)
    b0_tesla: float = Field(default=3.0, ge=0.5, le=11.7)
    fov_x_m: float = Field(default=0.24, gt=0.05, le=0.50)
    fov_y_m: float = Field(default=0.24, gt=0.05, le=0.50)
    noise_snr_db: float = Field(default=35.0, ge=10.0, le=60.0)
    mask_type: Literal["cartesian", "radial", "poisson"] = "cartesian"
    acceleration: int = Field(default=4, ge=1, le=16)
    acs_lines: int = Field(default=18, ge=4, le=64)


class TrainingStartConfig(BaseModel):
    """Hiperparametros fisicos y de red para iniciar el entrenamiento PINN por WebSocket."""

    b0_tesla: float = Field(default=3.0, ge=0.5, le=11.7)
    mask_type: Literal["cartesian", "radial", "poisson"] = "cartesian"
    acceleration: int = Field(default=4, ge=1, le=16)
    acs_lines: int = Field(default=18, ge=4, le=64)
    epochs: int = Field(default=80, ge=5, le=500)
    warmup_epochs: int = Field(default=15, ge=0, le=100)
    omega_anatomy: float = Field(default=30.0, ge=5.0, le=60.0)
    omega_coil: float = Field(default=5.0, ge=1.0, le=20.0)
    omega_ept: float = Field(default=15.0, ge=2.0, le=40.0)
    lr_anatomy: float = Field(default=7.0e-4, gt=1e-6, le=1e-2)
    lr_coil: float = Field(default=1.2e-3, gt=1e-6, le=1e-2)
    lr_ept: float = Field(default=2.5e-3, gt=1e-6, le=1e-2)
    lambda_kspace: float = Field(default=1.0, ge=0.01, le=20.0)
    lambda_ept_sigma: float = Field(default=0.15, ge=0.0, le=5.0)
    lambda_ept_eps: float = Field(default=0.02, ge=0.0, le=5.0)
    lambda_coil_smooth: float = Field(default=0.02, ge=0.0, le=2.0)
    lambda_harmonic_b0: float = Field(default=0.02, ge=0.0, le=2.0)
    telemetry_interval: int = Field(default=2, ge=1, le=25)


class DatasetStateResponse(BaseModel):
    """Estado actual del volumen en GPU y mapas iniciales codificados en Float32 Base64."""

    source: str
    height: int
    width: int
    num_coils: int
    b0_tesla: float
    larmor_freq_mhz: float
    fov_x_m: float
    fov_y_m: float
    mask_type: str
    acceleration: int
    acs_lines: int
    sampling_fraction: float
    zero_filled_psnr: float
    zero_filled_ssim: float
    gpu_name: str
    vram_allocated_mb: float
    maps_f32_b64: Dict[str, str]


class EpochTelemetryPayload(BaseModel):
    """Mensaje emitido por el WebSocket (/ws/train) en cada epoca."""

    type: Literal["epoch", "completed", "stopped", "error"]
    epoch: int
    total_epochs: int
    total_loss: float
    loss_kspace: float
    loss_helmholtz: float
    loss_coil: float
    loss_harmonic: float
    psnr_db: float
    ssim: float
    nrmse_sigma: float
    tumor_sigma_mean: float
    ept_active: bool
    elapsed_ms: float
    maps_f32_b64: Optional[Dict[str, str]] = None
    message: Optional[str] = None
