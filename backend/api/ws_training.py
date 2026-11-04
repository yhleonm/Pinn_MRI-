"""WebSocket Endpoint (/ws/train) for Real-Time GPU PINN Training and Tensor Streaming."""

from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.training.trainer import MRIPINNTrainer
from ..schemas import EpochTelemetryPayload, TrainingStartConfig
from .routes_kspace import gpu_session, tensor_to_f32_b64

router = APIRouter(tags=["websocket"])


def _build_trainer_config(cfg: TrainingStartConfig) -> Dict[str, Any]:
    """Estructura el diccionario de configuracion compatible con MRIPINNTrainer."""
    if gpu_session.dataset is None:
        gpu_session.initialize_default()
    assert gpu_session.dataset is not None

    return {
        "physics": {
            "b0_tesla": cfg.b0_tesla,
            "larmor_freq_mhz": round(42.58 * cfg.b0_tesla, 2),
            "fov_x_m": gpu_session.dataset.fov_x_m,
            "fov_y_m": gpu_session.dataset.fov_y_m,
            "epsilon_b1": 1e-3,
        },
        "data": {
            "mask_type": cfg.mask_type,
            "acceleration": cfg.acceleration,
            "acs_lines": cfg.acs_lines,
        },
        "model": {
            "omega_anatomy": cfg.omega_anatomy,
            "omega_coil": cfg.omega_coil,
            "omega_ept": cfg.omega_ept,
            "hidden_features_anatomy": 128,
            "hidden_layers_anatomy": 4,
            "hidden_features_coil": 64,
            "hidden_layers_coil": 3,
            "hidden_features_ept": 96,
            "hidden_layers_ept": 3,
            "sigma_bounds": [0.01, 3.0],
            "eps_r_bounds": [1.0, 100.0],
        },
        "training": {
            "epochs": cfg.epochs,
            "warmup_epochs": cfg.warmup_epochs,
            "lr_anatomy": cfg.lr_anatomy,
            "lr_coil": cfg.lr_coil,
            "lr_ept": cfg.lr_ept,
            "lambda_kspace": cfg.lambda_kspace,
            "lambda_ept_sigma": cfg.lambda_ept_sigma,
            "lambda_ept_eps": cfg.lambda_ept_eps,
            "lambda_coil_smooth": cfg.lambda_coil_smooth,
            "lambda_harmonic_b0": cfg.lambda_harmonic_b0,
            "telemetry_interval": cfg.telemetry_interval,
        },
    }


@router.websocket("/ws/train")
async def websocket_training_endpoint(websocket: WebSocket) -> None:
    """
    Canal WebSocket bidireccional para controlar el entrenamiento en la RTX 4060 Ti
    y transmitir tensores Float32 (|m|, sigma, eps_r, phi_tr) + metricas en vivo.
    """
    await websocket.accept()
    training_task: Optional[asyncio.Task[None]] = None

    async def run_training_loop(cfg: TrainingStartConfig) -> None:
        try:
            # Actualizar mascara y B0 si cambiaron desde el panel lateral
            gpu_session.update_mask_and_b0(
                mask_type=cfg.mask_type,
                acceleration=cfg.acceleration,
                acs_lines=cfg.acs_lines,
                b0_tesla=cfg.b0_tesla,
            )
            assert gpu_session.dataset is not None
            trainer_cfg = _build_trainer_config(cfg)
            trainer = MRIPINNTrainer(
                dataset=gpu_session.dataset,
                config=trainer_cfg,
                device=gpu_session.device,
            )
            gpu_session.trainer = trainer

            total_epochs = cfg.epochs
            interval = max(1, cfg.telemetry_interval)
            ds = gpu_session.dataset
            h_ds, w_ds = ds.sigma_gt.shape
            col_grid = torch.arange(w_ds, device=ds.sigma_gt.device).unsqueeze(0).expand(h_ds, w_ds)
            tumor_roi = (ds.sigma_gt >= 1.15) & (ds.sigma_gt <= 1.45) & (col_grid > int(w_ds * 0.54))
            if int(tumor_roi.sum().item()) == 0:
                tumor_roi = (ds.sigma_gt > 1.0) & (ds.sigma_gt < 1.45)

            for epoch_idx in range(total_epochs):
                if trainer.stop_requested:
                    break

                # Ejecutar paso de epoca en hilo trabajador para no bloquear el event loop de FastAPI
                metrics, maps = await asyncio.to_thread(trainer.step_epoch, epoch_idx)
                gpu_session.last_maps = maps

                include_maps = (
                    (epoch_idx + 1) == 1
                    or (epoch_idx + 1) == cfg.warmup_epochs
                    or (epoch_idx + 1) == cfg.warmup_epochs + 1
                    or (epoch_idx + 1) % interval == 0
                    or (epoch_idx + 1) == total_epochs
                )

                maps_payload: Optional[Dict[str, str]] = None
                if include_maps:
                    maps_payload = {
                        "m_mag": tensor_to_f32_b64(maps["m_mag"]),
                        "sigma": tensor_to_f32_b64(maps["sigma"]),
                        "eps_r": tensor_to_f32_b64(maps["eps_r"]),
                        "sar": tensor_to_f32_b64(maps["sar"]),
                        "phi_tr": tensor_to_f32_b64(maps["phi_tr"]),
                        "phi_b0": tensor_to_f32_b64(maps["phi_b0"]),
                    }

                if int(tumor_roi.sum().item()) > 0:
                    tumor_sig = float(maps["sigma"][tumor_roi].mean().item())
                else:
                    tumor_sig = float(maps["sigma"].max().item())

                is_last = (epoch_idx + 1) == total_epochs and not trainer.stop_requested
                payload = EpochTelemetryPayload(
                    type="completed" if is_last else "epoch",
                    epoch=metrics.epoch,
                    total_epochs=total_epochs,
                    total_loss=metrics.total_loss,
                    loss_kspace=metrics.loss_kspace,
                    loss_helmholtz=metrics.loss_helmholtz,
                    loss_coil=metrics.loss_coil,
                    loss_harmonic=metrics.loss_harmonic,
                    psnr_db=metrics.psnr_db,
                    ssim=metrics.ssim,
                    nrmse_sigma=metrics.nrmse_sigma,
                    tumor_sigma_mean=tumor_sig,
                    ept_active=metrics.ept_active,
                    elapsed_ms=metrics.elapsed_ms,
                    maps_f32_b64=maps_payload,
                )
                await websocket.send_json(payload.model_dump())
                await asyncio.sleep(0.005)

            if trainer.stop_requested and gpu_session.last_maps:
                last_m = trainer.history[-1]
                await websocket.send_json(
                    EpochTelemetryPayload(
                        type="stopped",
                        epoch=last_m.epoch,
                        total_epochs=total_epochs,
                        total_loss=last_m.total_loss,
                        loss_kspace=last_m.loss_kspace,
                        loss_helmholtz=last_m.loss_helmholtz,
                        loss_coil=last_m.loss_coil,
                        loss_harmonic=last_m.loss_harmonic,
                        psnr_db=last_m.psnr_db,
                        ssim=last_m.ssim,
                        nrmse_sigma=last_m.nrmse_sigma,
                        tumor_sigma_mean=float(gpu_session.last_maps["sigma"].max().item()),
                        ept_active=last_m.ept_active,
                        elapsed_ms=last_m.elapsed_ms,
                        message="Entrenamiento detenido por el usuario.",
                    ).model_dump()
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "epoch": 0,
                    "total_epochs": cfg.epochs,
                    "total_loss": 0.0,
                    "loss_kspace": 0.0,
                    "loss_helmholtz": 0.0,
                    "loss_coil": 0.0,
                    "loss_harmonic": 0.0,
                    "psnr_db": 0.0,
                    "ssim": 0.0,
                    "nrmse_sigma": 0.0,
                    "tumor_sigma_mean": 0.0,
                    "ept_active": False,
                    "elapsed_ms": 0.0,
                    "message": str(exc),
                }
            )

    try:
        while True:
            data = await websocket.receive_json()
            action = str(data.get("action", "")).lower()

            if action == "start":
                if training_task is not None and not training_task.done():
                    if gpu_session.trainer is not None:
                        gpu_session.trainer.stop_requested = True
                    training_task.cancel()
                    await asyncio.sleep(0.05)

                raw_cfg = data.get("config", {})
                cfg = TrainingStartConfig(**raw_cfg)
                training_task = asyncio.create_task(run_training_loop(cfg))

            elif action == "stop":
                if gpu_session.trainer is not None:
                    gpu_session.trainer.stop_requested = True
    except WebSocketDisconnect:
        if gpu_session.trainer is not None:
            gpu_session.trainer.stop_requested = True
        if training_task is not None and not training_task.done():
            training_task.cancel()
