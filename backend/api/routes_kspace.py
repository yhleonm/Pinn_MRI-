"""REST Endpoints for K-Space management, .h5 upload, subsampling masks, and Zero-Filled baseline."""

from __future__ import annotations
import base64
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.data.kspace_loader import (
    SyntheticPhantomDataset,
    generate_clinical_in_vivo_brain_phantom,
    generate_realistic_brain_ept_phantom,
    generate_shepp_logan_ept_phantom,
    load_h5_kspace_slice,
    save_dataset_to_h5,
)
from src.data.masks import create_sampling_mask
from src.physics.helmholtz_ept import HelmholtzEPTOperator
from src.physics.sense_operator import DifferentiableSENSE
from src.training.trainer import MRIPINNTrainer, compute_psnr, compute_ssim
from ..schemas import DatasetStateResponse, PhantomConfigRequest

router = APIRouter(prefix="/api", tags=["kspace"])


def tensor_to_f32_b64(tensor: torch.Tensor) -> str:
    """
    Convierte un tensor 2D de PyTorch a arreglo binario IEEE-754 float32 (little-endian)
    codificado en Base64 para decodificacion instantanea en Float32Array del navegador.
    """
    arr = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    return base64.b64encode(arr.tobytes()).decode("ascii")


class GPUSessionManager:
    """Gestor persistente en VRAM de la RTX 4060 Ti para el espacio-k y el modelo PINN."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sense_op = DifferentiableSENSE().to(self.device)
        self.dataset: Optional[SyntheticPhantomDataset] = None
        self.source: str = "synthetic_shepp_logan_8coils"
        self.mask_type: str = "cartesian"
        self.acceleration: int = 4
        self.acs_lines: int = 18
        self.noise_snr_db: float = 35.0
        self.trainer: Optional[MRIPINNTrainer] = None
        self.last_maps: Dict[str, torch.Tensor] = {}

    def initialize_default(self) -> None:
        """Inicializa el phantom analitico Shepp-Logan multi-bobina en GPU al arrancar."""
        self.dataset = generate_shepp_logan_ept_phantom(
            height=128,
            width=128,
            num_coils=8,
            fov_x_m=0.24,
            fov_y_m=0.24,
            b0_tesla=3.0,
            noise_snr_db=35.0,
            mask_type="cartesian",
            acceleration=4,
            acs_lines=18,
            device=self.device,
        )
        cortical_ds = generate_realistic_brain_ept_phantom(
            height=128,
            width=128,
            num_coils=8,
            fov_x_m=0.24,
            fov_y_m=0.24,
            b0_tesla=3.0,
            noise_snr_db=36.0,
            mask_type="cartesian",
            acceleration=4,
            acs_lines=18,
            device=self.device,
        )
        cortical_path = Path("data/cortical_brain_tumor_8coils_3T.h5")
        save_dataset_to_h5(cortical_ds, cortical_path)
        sample_path = Path("data/raw/sample_brain_8coils.h5")
        save_dataset_to_h5(cortical_ds, sample_path)
        clinical_path = Path("data/clinical_in_vivo_brain_8coils.h5")
        if not clinical_path.exists():
            clinical_ds = generate_clinical_in_vivo_brain_phantom(device=self.device)
            save_dataset_to_h5(clinical_ds, clinical_path)
        shepp_path = Path("data/shepp_logan_8coils_3T.h5")
        if not shepp_path.exists():
            save_dataset_to_h5(self.dataset, shepp_path)

    def update_mask_and_b0(
        self,
        mask_type: str,
        acceleration: int,
        acs_lines: int,
        b0_tesla: float,
    ) -> None:
        """Actualiza dinamicamente la mascara P(k), el campo B0 y la reconstruccion Zero-Filled."""
        if self.dataset is None:
            self.initialize_default()
        assert self.dataset is not None

        _, h, w = self.dataset.kspace_full.shape
        old_b0 = float(self.dataset.b0_tesla)
        self.mask_type = mask_type
        self.acceleration = acceleration
        self.acs_lines = acs_lines
        self.dataset.b0_tesla = b0_tesla

        if abs(b0_tesla - old_b0) > 1e-3 and old_b0 > 0.1:
            scale_b0 = float(b0_tesla) / old_b0
            self.dataset.phi_tr_gt = self.dataset.phi_tr_gt * scale_b0
            self.dataset.phi_total_gt = self.dataset.phi_tr_gt + self.dataset.phi_b0_gt
            m_cplx_new = torch.polar(self.dataset.m_mag_gt, self.dataset.phi_total_gt)
            self.dataset.kspace_full = self.sense_op.forward(
                m_cplx_new, self.dataset.coil_maps_gt, mask=None
            )

        mask_t = create_sampling_mask(
            height=h,
            width=w,
            mask_type=mask_type,
            acceleration=acceleration,
            acs_lines=acs_lines,
            device=self.device,
        )
        kspace_masked = self.dataset.kspace_full * mask_t.unsqueeze(0)
        coil_est = self.sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=acs_lines)
        zero_filled = self.sense_op.adjoint(kspace_masked, coil_maps=coil_est)

        self.dataset.mask = mask_t
        self.dataset.kspace_masked = kspace_masked
        self.dataset.zero_filled = zero_filled

    def build_state_response(self) -> DatasetStateResponse:
        if self.dataset is None:
            self.initialize_default()
        assert self.dataset is not None

        ds = self.dataset
        nc, h, w = ds.kspace_full.shape
        zf_mag = torch.abs(ds.zero_filled)
        zf_psnr = compute_psnr(zf_mag, ds.m_mag_gt)
        zf_ssim = compute_ssim(zf_mag, ds.m_mag_gt)

        # Espectro log-k-space combinado por suma de magnitudes de bobinas: log(1 + |s(k)|)
        k_full_mag = torch.sqrt(torch.sum(torch.abs(ds.kspace_full) ** 2, dim=0))
        k_log_full = torch.log1p(k_full_mag * 25.0)
        k_masked_mag = torch.sqrt(torch.sum(torch.abs(ds.kspace_masked) ** 2, dim=0))
        k_log_masked = torch.log1p(k_masked_mag * 25.0)

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        vram_mb = (
            torch.cuda.memory_allocated(0) / (1024.0 * 1024.0)
            if self.device.type == "cuda"
            else 0.0
        )

        # Si ya hubo entrenamiento previo, incluir los ultimos mapas predichos; de lo contrario, inicializar con Zero-Filled / GT
        m_pred = self.last_maps.get("m_mag", zf_mag)
        sigma_pred = self.last_maps.get("sigma", ds.sigma_gt * 0.0)
        eps_pred = self.last_maps.get("eps_r", ds.eps_r_gt * 0.0)
        phi_tr_pred = self.last_maps.get("phi_tr", torch.angle(ds.zero_filled))
        sar_gt = (
            HelmholtzEPTOperator.compute_local_sar(
                sigma=ds.sigma_gt,
                b1_mag=ds.m_mag_gt,
                coords_norm=None,
                b0_tesla=ds.b0_tesla,
            )
            * ds.tissue_mask_gt
        )
        sar_pred = self.last_maps.get("sar", sar_gt * 0.0)

        maps_f32_b64 = {
            "zero_filled_mag": tensor_to_f32_b64(zf_mag),
            "m_mag": tensor_to_f32_b64(m_pred),
            "sigma": tensor_to_f32_b64(sigma_pred),
            "eps_r": tensor_to_f32_b64(eps_pred),
            "sar": tensor_to_f32_b64(sar_pred),
            "phi_tr": tensor_to_f32_b64(phi_tr_pred),
            "m_mag_gt": tensor_to_f32_b64(ds.m_mag_gt),
            "sigma_gt": tensor_to_f32_b64(ds.sigma_gt),
            "eps_r_gt": tensor_to_f32_b64(ds.eps_r_gt),
            "sar_gt": tensor_to_f32_b64(sar_gt),
            "phi_tr_gt": tensor_to_f32_b64(ds.phi_tr_gt),
            "phi_b0_gt": tensor_to_f32_b64(ds.phi_b0_gt),
            "kspace_log_full": tensor_to_f32_b64(k_log_full),
            "kspace_log_masked": tensor_to_f32_b64(k_log_masked),
            "mask": tensor_to_f32_b64(ds.mask),
            "tissue_mask": tensor_to_f32_b64(ds.tissue_mask_gt),
        }

        sampling_frac = float(torch.mean(ds.mask).item())
        return DatasetStateResponse(
            source=self.source,
            height=h,
            width=w,
            num_coils=nc,
            b0_tesla=ds.b0_tesla,
            larmor_freq_mhz=round(42.58 * ds.b0_tesla, 2),
            fov_x_m=ds.fov_x_m,
            fov_y_m=ds.fov_y_m,
            mask_type=self.mask_type,
            acceleration=self.acceleration,
            acs_lines=self.acs_lines,
            sampling_fraction=sampling_frac,
            zero_filled_psnr=zf_psnr,
            zero_filled_ssim=zf_ssim,
            gpu_name=gpu_name,
            vram_allocated_mb=round(vram_mb, 2),
            maps_f32_b64=maps_f32_b64,
        )


gpu_session = GPUSessionManager()


@router.get("/state", response_model=DatasetStateResponse)
async def get_current_state() -> DatasetStateResponse:
    """Devuelve el estado actual del espacio-k en GPU, metricas base y mapas Float32."""
    return gpu_session.build_state_response()


@router.post("/phantom", response_model=DatasetStateResponse)
async def generate_or_update_phantom(req: PhantomConfigRequest) -> DatasetStateResponse:
    """Genera un nuevo Phantom Analitico Shepp-Logan Multi-Bobina con la configuracion dada."""
    gpu_session.mask_type = req.mask_type
    gpu_session.acceleration = req.acceleration
    gpu_session.acs_lines = req.acs_lines
    gpu_session.noise_snr_db = req.noise_snr_db
    gpu_session.source = f"synthetic_shepp_logan_{req.num_coils}ch"
    gpu_session.last_maps.clear()

    gpu_session.dataset = generate_shepp_logan_ept_phantom(
        height=req.height,
        width=req.width,
        num_coils=req.num_coils,
        fov_x_m=req.fov_x_m,
        fov_y_m=req.fov_y_m,
        b0_tesla=req.b0_tesla,
        noise_snr_db=req.noise_snr_db,
        mask_type=req.mask_type,
        acceleration=req.acceleration,
        acs_lines=req.acs_lines,
        device=gpu_session.device,
    )
    return gpu_session.build_state_response()


@router.post("/upload-h5", response_model=DatasetStateResponse)
async def upload_h5_file(
    file: UploadFile = File(...),
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    b0_tesla: float = 3.0,
) -> DatasetStateResponse:
    """Recibe un archivo .h5 (fastMRI / Calgary-Campinas), lo guarda en data/raw/ y lo carga en GPU."""
    if not file.filename or not file.filename.endswith((".h5", ".hdf5")):
        raise HTTPException(status_code=400, detail="El archivo debe tener extension .h5 o .hdf5")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / Path(file.filename).name

    content = await file.read()
    dest_path.write_bytes(content)

    try:
        gpu_session.dataset = load_h5_kspace_slice(
            h5_path=dest_path,
            mask_type=mask_type,
            acceleration=acceleration,
            acs_lines=acs_lines,
            b0_tesla=b0_tesla,
            device=gpu_session.device,
        )
        gpu_session.source = dest_path.name
        gpu_session.mask_type = mask_type
        gpu_session.acceleration = acceleration
        gpu_session.acs_lines = acs_lines
        gpu_session.last_maps.clear()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Error al procesar .h5: {exc}") from exc

    return gpu_session.build_state_response()


@router.post("/load-sample-h5", response_model=DatasetStateResponse)
async def load_sample_h5(req: PhantomConfigRequest) -> DatasetStateResponse:
    """Carga el corte anatomico cortical realista .h5 en data/cortical_brain_tumor_8coils_3T.h5."""
    sample_path = Path("data/cortical_brain_tumor_8coils_3T.h5")
    if not sample_path.exists():
        gpu_session.initialize_default()

    gpu_session.dataset = load_h5_kspace_slice(
        h5_path=sample_path,
        mask_type=req.mask_type,
        acceleration=req.acceleration,
        acs_lines=req.acs_lines,
        fov_x_m=req.fov_x_m,
        fov_y_m=req.fov_y_m,
        b0_tesla=req.b0_tesla,
        device=gpu_session.device,
    )
    gpu_session.source = sample_path.name
    gpu_session.mask_type = req.mask_type
    gpu_session.acceleration = req.acceleration
    gpu_session.acs_lines = req.acs_lines
    gpu_session.last_maps.clear()
    return gpu_session.build_state_response()
