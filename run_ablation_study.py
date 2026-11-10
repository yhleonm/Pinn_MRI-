"""
Automated GPU Benchmark Suite, Classical Baseline Comparisons (CG-SENSE, Savitzky-Golay EPT, cr-MREPT),
Multi-Seed Statistical Evaluation, Confounding Controls, and Print-Ready White-Background 300-DPI Figure
Generator for Decoupled MRI-PINN (RACCEFYN).
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import torch

from src.data.kspace_loader import (
    generate_clinical_in_vivo_brain_phantom,
    generate_realistic_brain_ept_phantom,
    generate_shepp_logan_ept_phantom,
    load_h5_kspace_slice,
    save_dataset_to_h5,
)
from src.physics.helmholtz_ept import HelmholtzEPTOperator
from src.training.trainer import (
    MRIPINNTrainer,
    compute_cg_sense,
    compute_convection_reaction_ept,
    compute_nrmse,
    compute_psnr,
    compute_savitzky_golay_ept,
    compute_ssim,
    separate_harmonic_b0_lbv,
    unwrap_phase_2d,
)


def _base_config(
    b0_tesla: float = 3.0,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    epochs: int = 55,
    lambda_ept_sigma: float = 0.15,
    lambda_harmonic_b0: float = 0.02,
    enable_s1_unwrapping: bool = True,
) -> Dict[str, Any]:
    return {
        "physics": {
            "b0_tesla": b0_tesla,
            "larmor_freq_mhz": round(42.58 * b0_tesla, 2),
            "fov_x_m": 0.24,
            "fov_y_m": 0.24,
            "epsilon_b1": 1e-3,
        },
        "data": {
            "mask_type": mask_type,
            "acceleration": acceleration,
            "acs_lines": 18,
        },
        "model": {
            "omega_anatomy": 30.0,
            "omega_coil": 5.0,
            "omega_ept": 15.0,
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
            "epochs": epochs,
            "warmup_epochs": 15,
            "lr_anatomy": 7.0e-4,
            "lr_coil": 1.2e-3,
            "lr_ept": 2.5e-3,
            "lambda_kspace": 1.0,
            "lambda_ept_sigma": lambda_ept_sigma,
            "lambda_ept_eps": 0.02,
            "lambda_coil_smooth": 0.02,
            "lambda_harmonic_b0": lambda_harmonic_b0,
            "enable_s1_unwrapping": enable_s1_unwrapping,
            "telemetry_interval": 15,
        },
    }


def _get_tumor_and_healthy_masks(ds: Any) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = ds.sigma_gt.shape
    device = ds.sigma_gt.device
    y = torch.linspace(-1.0, 1.0, steps=h, device=device)
    x = torch.linspace(-1.0, 1.0, steps=w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    tumor_mask = (((xx - 0.36) / 0.115) ** 2 + ((yy + 0.16) / 0.110) ** 2) <= 1.0
    healthy_mask = (ds.tissue_mask_gt > 0.5) & (~tumor_mask) & (ds.sigma_gt < 0.70) & (ds.sigma_gt > 0.22)
    return tumor_mask, healthy_mask


def _extract_roi_metrics(ds: Any, maps: Dict[str, torch.Tensor]) -> Dict[str, float]:
    tumor_mask, healthy_mask = _get_tumor_and_healthy_masks(ds)

    sigma_pred = maps["sigma"]
    eps_pred = maps["eps_r"]
    sar_pred = maps["sar"]
    sar_gt = (
        HelmholtzEPTOperator.compute_local_sar(
            sigma=ds.sigma_gt,
            b1_mag=ds.m_mag_gt,
            coords_norm=None,
            b0_tesla=ds.b0_tesla,
        )
        * ds.tissue_mask_gt
    )

    tumor_sig_pred = float(sigma_pred[tumor_mask].mean().item()) if int(tumor_mask.sum()) > 0 else 0.0
    tumor_sig_gt = float(ds.sigma_gt[tumor_mask].mean().item()) if int(tumor_mask.sum()) > 0 else 1.22
    healthy_sig_pred = float(sigma_pred[healthy_mask].mean().item()) if int(healthy_mask.sum()) > 0 else 0.0
    tumor_eps_pred = float(eps_pred[tumor_mask].mean().item()) if int(tumor_mask.sum()) > 0 else 0.0
    tumor_eps_gt = float(ds.eps_r_gt[tumor_mask].mean().item()) if int(tumor_mask.sum()) > 0 else 68.0
    tumor_sar_pred = float(sar_pred[tumor_mask].mean().item()) if int(tumor_mask.sum()) > 0 else 0.0
    tumor_sar_gt = float(sar_gt[tumor_mask].mean().item()) if int(tumor_mask.sum()) > 0 else 0.0
    healthy_sar_pred = float(sar_pred[healthy_mask].mean().item()) if int(healthy_mask.sum()) > 0 else 0.0
    peak_sar_pred = float(sar_pred[ds.tissue_mask_gt > 0.5].max().item())

    return {
        "tumor_sigma_pred": tumor_sig_pred,
        "tumor_sigma_gt": tumor_sig_gt,
        "healthy_sigma_pred": healthy_sig_pred,
        "tumor_eps_pred": tumor_eps_pred,
        "tumor_eps_gt": tumor_eps_gt,
        "tumor_sar_pred": tumor_sar_pred,
        "tumor_sar_gt": tumor_sar_gt,
        "healthy_sar_pred": healthy_sar_pred,
        "peak_sar_pred": peak_sar_pred,
    }


def _add_scale_bar_5cm(ax: Any, width_px: int = 128, fov_m: float = 0.24) -> None:
    """Agrega una barra de escala fisica de 5 cm (26.7 pixeles) en la esquina inferior derecha."""
    bar_len_px = (0.05 / fov_m) * width_px
    x0 = width_px - bar_len_px - 8
    y0 = width_px - 10
    rect = Rectangle((x0, y0), bar_len_px, 2.5, facecolor="white", edgecolor="black", linewidth=0.5)
    ax.add_patch(rect)
    ax.text(
        x0 + bar_len_px / 2.0,
        y0 - 3.5,
        "5 cm",
        color="white",
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.55, edgecolor="none"),
    )


def main() -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/paper_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(" SUITE DE VALIDACION NUMERICA, COMPARACION CON METODOS CLASICOS Y FIGURAS BLANCAS (300 DPI)")
    print(f" Dispositivo: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 96)

    # 1. Generar fantomas numericos .h5 (con nombres actualizados y alias de compatibilidad)
    cortical_h5 = Path("data/cortical_brain_tumor_8coils_3T.h5")
    numerical_h5 = Path("data/numerical_brain_phantom_8coils.h5")
    clinical_alias_h5 = Path("data/clinical_in_vivo_brain_8coils.h5")
    shepp_h5 = Path("data/shepp_logan_8coils_3T.h5")

    cortical_ds_base = generate_realistic_brain_ept_phantom(device=device)
    save_dataset_to_h5(cortical_ds_base, cortical_h5)
    save_dataset_to_h5(cortical_ds_base, Path("data/raw/sample_brain_8coils.h5"))

    numerical_ds_base = generate_clinical_in_vivo_brain_phantom(device=device)
    save_dataset_to_h5(numerical_ds_base, numerical_h5)
    save_dataset_to_h5(numerical_ds_base, clinical_alias_h5)

    shepp_ds_base = generate_shepp_logan_ept_phantom(device=device)
    save_dataset_to_h5(shepp_ds_base, shepp_h5)

    epochs = 55
    seeds = [108, 202, 305, 409, 512]
    results: Dict[str, Any] = {
        "multifield_numerical": [],
        "trajectories_stress": [],
        "ablation_and_baselines": [],
        "confounding_controls": [],
        "snr_sensitivity": [],
    }

    # -------------------------------------------------------------------------
    # ESTUDIO 1: Multi-Campo (1.5T, 3.0T, 7.0T) con N=5 semillas sobre Fantoma Numerico .h5
    # -------------------------------------------------------------------------
    print("\n[Estudio 1/5] Evaluando Multi-Campo (1.5T, 3.0T, 7.0T) con N=5 semillas (R=4x Radial, 25% k-space)...")
    multifield_maps: Dict[float, Dict[str, np.ndarray]] = {}
    for b0 in [1.5, 3.0, 7.0]:
        seed_runs: List[Dict[str, float]] = []
        for s_idx, sd in enumerate(seeds):
            torch.manual_seed(sd)
            np.random.seed(sd)
            ds_s = generate_clinical_in_vivo_brain_phantom(
                b0_tesla=b0,
                mask_type="radial",
                acceleration=4,
                acs_lines=18,
                seed=sd,
                device=device,
            )
            cfg = _base_config(b0_tesla=b0, mask_type="radial", acceleration=4, epochs=epochs)
            trainer = MRIPINNTrainer(dataset=ds_s, config=cfg, device=device)
            history, maps = trainer.train(epochs=epochs, verbose=False)
            last = history[-1]

            zf_psnr = compute_psnr(torch.abs(ds_s.zero_filled), ds_s.m_mag_gt)
            zf_ssim = compute_ssim(torch.abs(ds_s.zero_filled), ds_s.m_mag_gt)
            cg_cplx = compute_cg_sense(ds_s.kspace_masked, trainer.coil_acs_ref, ds_s.mask)
            cg_psnr = compute_psnr(torch.abs(cg_cplx), ds_s.m_mag_gt)
            cg_ssim = compute_ssim(torch.abs(cg_cplx), ds_s.m_mag_gt)
            roi = _extract_roi_metrics(ds_s, maps)

            seed_runs.append(
                {
                    "zf_psnr": zf_psnr,
                    "zf_ssim": zf_ssim,
                    "cg_psnr": cg_psnr,
                    "cg_ssim": cg_ssim,
                    "pinn_psnr": last.psnr_db,
                    "pinn_ssim": last.ssim,
                    "nrmse_sigma": last.nrmse_sigma,
                    "nrmse_eps": last.nrmse_eps,
                    **roi,
                }
            )
            if s_idx == 0:
                multifield_maps[b0] = {
                    "m_mag": maps["m_mag"].detach().cpu().numpy(),
                    "phi_tr": (maps["phi_tr"] * ds_s.tissue_mask_gt).detach().cpu().numpy(),
                    "sigma": maps["sigma"].detach().cpu().numpy(),
                    "eps_r": maps["eps_r"].detach().cpu().numpy(),
                    "sar": maps["sar"].detach().cpu().numpy(),
                    "sigma_gt": ds_s.sigma_gt.detach().cpu().numpy(),
                    "eps_r_gt": ds_s.eps_r_gt.detach().cpu().numpy(),
                    "sigma_err": torch.abs((maps["sigma"] - ds_s.sigma_gt) * ds_s.tissue_mask_gt).detach().cpu().numpy(),
                }

        def _stat(k: str) -> tuple[float, float]:
            vals = [r[k] for r in seed_runs]
            return float(np.mean(vals)), float(np.std(vals, ddof=1))

        pinn_psnr_m, pinn_psnr_s = _stat("pinn_psnr")
        pinn_ssim_m, pinn_ssim_s = _stat("pinn_ssim")
        zf_psnr_m, _ = _stat("zf_psnr")
        cg_psnr_m, _ = _stat("cg_psnr")
        nrmse_sig_m, nrmse_sig_s = _stat("nrmse_sigma")
        nrmse_eps_m, nrmse_eps_s = _stat("nrmse_eps")
        tsig_m, tsig_s = _stat("tumor_sigma_pred")
        teps_m, teps_s = _stat("tumor_eps_pred")
        tsar_m, tsar_s = _stat("tumor_sar_pred")
        hsar_m, hsar_s = _stat("healthy_sar_pred")

        entry = {
            "b0_tesla": b0,
            "larmor_mhz": round(42.58 * b0, 2),
            "zf_psnr": zf_psnr_m,
            "cg_psnr": cg_psnr_m,
            "pinn_psnr": pinn_psnr_m,
            "pinn_psnr_std": pinn_psnr_s,
            "pinn_ssim": pinn_ssim_m,
            "pinn_ssim_std": pinn_ssim_s,
            "nrmse_sigma": nrmse_sig_m,
            "nrmse_sigma_std": nrmse_sig_s,
            "nrmse_eps": nrmse_eps_m,
            "nrmse_eps_std": nrmse_eps_s,
            "tumor_sigma_pred": tsig_m,
            "tumor_sigma_std": tsig_s,
            "tumor_sigma_gt": seed_runs[0]["tumor_sigma_gt"],
            "tumor_eps_pred": teps_m,
            "tumor_eps_std": teps_s,
            "tumor_eps_gt": seed_runs[0]["tumor_eps_gt"],
            "healthy_sar_pred": hsar_m,
            "healthy_sar_std": hsar_s,
            "tumor_sar_pred": tsar_m,
            "tumor_sar_std": tsar_s,
        }
        results["multifield_numerical"].append(entry)
        print(
            f"  * B0={b0:.1f}T ({entry['larmor_mhz']} MHz) | ZF: {zf_psnr_m:.2f} | CG-SENSE: {cg_psnr_m:.2f} | "
            f"PINN: {pinn_psnr_m:.2f}±{pinn_psnr_s:.2f} dB | SSIM: {pinn_ssim_m:.4f} | "
            f"NRMSE_sig: {nrmse_sig_m:.4f}±{nrmse_sig_s:.4f} (ROI {tsig_m:.2f}/{entry['tumor_sigma_gt']:.2f} S/m) | "
            f"NRMSE_eps: {nrmse_eps_m:.4f} (ROI {teps_m:.1f}/{entry['tumor_eps_gt']:.1f}) | "
            f"SAR Tumor: {tsar_m:.2f} W/kg (Sano: {hsar_m:.2f} W/kg)"
        )

    # -------------------------------------------------------------------------
    # ESTUDIO 2: Comparacion con Referencias Clasicas (CG-SENSE, SG-EPT, cr-EPT) y Ablaciones
    # -------------------------------------------------------------------------
    print("\n[Estudio 2/5] Evaluando Referencias Clasicas (CG-SENSE, SG-EPT, cr-MREPT) y Ablaciones (3.0T y 7.0T)...")
    torch.manual_seed(108)
    np.random.seed(108)
    ds_3t = generate_clinical_in_vivo_brain_phantom(
        b0_tesla=3.0, noise_snr_db=33.0, mask_type="poisson", acceleration=4, seed=108, device=device
    )
    torch.manual_seed(108)
    np.random.seed(108)
    ds_7t = generate_clinical_in_vivo_brain_phantom(
        b0_tesla=7.0, noise_snr_db=33.0, mask_type="poisson", acceleration=4, seed=108, device=device
    )
    tmask_3t, hmask_3t = _get_tumor_and_healthy_masks(ds_3t)
    tmask_7t, hmask_7t = _get_tumor_and_healthy_masks(ds_7t)

    # 1. Propuesta Completa @ 3.0T
    torch.manual_seed(108)
    np.random.seed(108)
    cfg_full_3t = _base_config(b0_tesla=3.0, mask_type="poisson", acceleration=4, epochs=epochs)
    tr_full_3t = MRIPINNTrainer(dataset=ds_3t, config=cfg_full_3t, device=device)
    h_full_3t, m_full_3t = tr_full_3t.train(epochs=epochs, verbose=False)
    roi_full_3t = _extract_roi_metrics(ds_3t, m_full_3t)

    # 2. Propuesta Completa @ 7.0T
    torch.manual_seed(108)
    np.random.seed(108)
    cfg_full_7t = _base_config(b0_tesla=7.0, mask_type="poisson", acceleration=4, epochs=epochs)
    tr_full_7t = MRIPINNTrainer(dataset=ds_7t, config=cfg_full_7t, device=device)
    h_full_7t, m_full_7t = tr_full_7t.train(epochs=epochs, verbose=False)
    roi_full_7t = _extract_roi_metrics(ds_7t, m_full_7t)

    # 3. Referencia 1: Adjunto Zero-Filled + Helmholtz-EPT Savitzky-Golay (ZF + SG-EPT) @ 3.0T
    zf_mag_3t = torch.abs(ds_3t.zero_filled)
    zf_psnr_3t = compute_psnr(zf_mag_3t, ds_3t.m_mag_gt)
    zf_ssim_3t = compute_ssim(zf_mag_3t, ds_3t.m_mag_gt)
    zf_phi_unw = unwrap_phase_2d(torch.angle(ds_3t.zero_filled), mag=zf_mag_3t)
    sig_zf_sg, eps_zf_sg = compute_savitzky_golay_ept(
        zf_phi_unw, zf_mag_3t, ds_3t.tissue_mask_gt, b0_tesla=3.0, window_size=7
    )
    sar_zf_sg = (
        HelmholtzEPTOperator.compute_local_sar(sig_zf_sg, zf_mag_3t, b0_tesla=3.0) * ds_3t.tissue_mask_gt
    )
    nrmse_sig_zf_sg = compute_nrmse(sig_zf_sg, ds_3t.sigma_gt, mask=ds_3t.tissue_mask_gt)
    nrmse_eps_zf_sg = compute_nrmse(eps_zf_sg, ds_3t.eps_r_gt, mask=ds_3t.tissue_mask_gt)
    tsig_zf_sg = float(sig_zf_sg[tmask_3t].mean().item())
    teps_zf_sg = float(eps_zf_sg[tmask_3t].mean().item())
    hsar_zf_sg = float(sar_zf_sg[hmask_3t].mean().item())
    tsar_zf_sg = float(sar_zf_sg[tmask_3t].mean().item())

    # 4. Referencia 2: CG-SENSE + Helmholtz-EPT Savitzky-Golay (CG-SENSE + SG-EPT) @ 3.0T
    cg_3t = compute_cg_sense(ds_3t.kspace_masked, tr_full_3t.coil_acs_ref, ds_3t.mask)
    cg_mag_3t = torch.abs(cg_3t)
    cg_psnr_3t = compute_psnr(cg_mag_3t, ds_3t.m_mag_gt)
    cg_ssim_3t = compute_ssim(cg_mag_3t, ds_3t.m_mag_gt)
    cg_phi_unw = unwrap_phase_2d(torch.angle(cg_3t), mag=cg_mag_3t)
    sig_cg_sg, eps_cg_sg = compute_savitzky_golay_ept(
        cg_phi_unw, cg_mag_3t, ds_3t.tissue_mask_gt, b0_tesla=3.0, window_size=7
    )
    sar_cg_sg = (
        HelmholtzEPTOperator.compute_local_sar(sig_cg_sg, cg_mag_3t, b0_tesla=3.0) * ds_3t.tissue_mask_gt
    )
    nrmse_sig_cg_sg = compute_nrmse(sig_cg_sg, ds_3t.sigma_gt, mask=ds_3t.tissue_mask_gt)
    nrmse_eps_cg_sg = compute_nrmse(eps_cg_sg, ds_3t.eps_r_gt, mask=ds_3t.tissue_mask_gt)
    tsig_cg_sg = float(sig_cg_sg[tmask_3t].mean().item())
    teps_cg_sg = float(eps_cg_sg[tmask_3t].mean().item())
    hsar_cg_sg = float(sar_cg_sg[hmask_3t].mean().item())
    tsar_cg_sg = float(sar_cg_sg[tmask_3t].mean().item())

    # 5. Referencia 3: CG-SENSE + cr-MREPT (Conveccion-Reaccion, Hafalir 2014) @ 3.0T
    sig_cg_cr = compute_convection_reaction_ept(
        cg_phi_unw, cg_mag_3t, ds_3t.tissue_mask_gt, b0_tesla=3.0
    )
    sar_cg_cr = (
        HelmholtzEPTOperator.compute_local_sar(sig_cg_cr, cg_mag_3t, b0_tesla=3.0) * ds_3t.tissue_mask_gt
    )
    nrmse_sig_cg_cr = compute_nrmse(sig_cg_cr, ds_3t.sigma_gt, mask=ds_3t.tissue_mask_gt)
    tsig_cg_cr = float(sig_cg_cr[tmask_3t].mean().item())
    hsar_cg_cr = float(sar_cg_cr[hmask_3t].mean().item())
    tsar_cg_cr = float(sar_cg_cr[tmask_3t].mean().item())

    # 6. Ablacion A: PINN-SENSE (lambda_sigma=0) + Savitzky-Golay EPT post-hoc sobre fase PINN @ 3.0T
    torch.manual_seed(108)
    np.random.seed(108)
    cfg_no_ept = _base_config(b0_tesla=3.0, mask_type="poisson", acceleration=4, epochs=epochs, lambda_ept_sigma=0.0)
    tr_no_ept = MRIPINNTrainer(dataset=ds_3t, config=cfg_no_ept, device=device)
    h_no_ept, m_no_ept = tr_no_ept.train(epochs=epochs, verbose=False)
    sig_pinn_sg, eps_pinn_sg = compute_savitzky_golay_ept(
        m_no_ept["phi_tr"], m_no_ept["m_mag"], ds_3t.tissue_mask_gt, b0_tesla=3.0, window_size=7
    )
    sar_pinn_sg = (
        HelmholtzEPTOperator.compute_local_sar(sig_pinn_sg, m_no_ept["m_mag"], b0_tesla=3.0) * ds_3t.tissue_mask_gt
    )
    nrmse_sig_pinn_sg = compute_nrmse(sig_pinn_sg, ds_3t.sigma_gt, mask=ds_3t.tissue_mask_gt)
    nrmse_eps_pinn_sg = compute_nrmse(eps_pinn_sg, ds_3t.eps_r_gt, mask=ds_3t.tissue_mask_gt)
    tsig_pinn_sg = float(sig_pinn_sg[tmask_3t].mean().item())
    teps_pinn_sg = float(eps_pinn_sg[tmask_3t].mean().item())
    hsar_pinn_sg = float(sar_pinn_sg[hmask_3t].mean().item())
    tsar_pinn_sg = float(sar_pinn_sg[tmask_3t].mean().item())

    # 7. Ablacion B: Decoupled MRI-PINN sin desenvolvimiento en S^1 a 7.0T (sobre el mismo ds_7t)
    zf_psnr_7t = compute_psnr(torch.abs(ds_7t.zero_filled), ds_7t.m_mag_gt)
    cg_7t = compute_cg_sense(ds_7t.kspace_masked, tr_full_7t.coil_acs_ref, ds_7t.mask)
    cg_psnr_7t = compute_psnr(torch.abs(cg_7t), ds_7t.m_mag_gt)
    torch.manual_seed(108)
    np.random.seed(108)
    cfg_no_s1_7t = _base_config(
        b0_tesla=7.0, mask_type="poisson", acceleration=4, epochs=epochs, enable_s1_unwrapping=False
    )
    tr_no_s1_7t = MRIPINNTrainer(dataset=ds_7t, config=cfg_no_s1_7t, device=device)
    h_no_s1_7t, m_no_s1_7t = tr_no_s1_7t.train(epochs=epochs, verbose=False)
    roi_no_s1_7t = _extract_roi_metrics(ds_7t, m_no_s1_7t)

    results["ablation_and_baselines"] = [
        {
            "method": "Adjunto Zero-Filled + SG-EPT ($7\\times 7$)",
            "b0_tesla": 3.0,
            "zf_psnr": zf_psnr_3t,
            "cg_psnr": cg_psnr_3t,
            "psnr_db": zf_psnr_3t,
            "ssim": zf_ssim_3t,
            "nrmse_sigma": nrmse_sig_zf_sg,
            "nrmse_eps": nrmse_eps_zf_sg,
            "tumor_sigma": tsig_zf_sg,
            "tumor_sigma_gt": roi_full_3t["tumor_sigma_gt"],
            "tumor_eps": teps_zf_sg,
            "tumor_eps_gt": roi_full_3t["tumor_eps_gt"],
            "healthy_sar": hsar_zf_sg,
            "tumor_sar": tsar_zf_sg,
        },
        {
            "method": "CG-SENSE + Savitzky-Golay EPT",
            "b0_tesla": 3.0,
            "zf_psnr": zf_psnr_3t,
            "cg_psnr": cg_psnr_3t,
            "psnr_db": cg_psnr_3t,
            "ssim": cg_ssim_3t,
            "nrmse_sigma": nrmse_sig_cg_sg,
            "nrmse_eps": nrmse_eps_cg_sg,
            "tumor_sigma": tsig_cg_sg,
            "tumor_sigma_gt": roi_full_3t["tumor_sigma_gt"],
            "tumor_eps": teps_cg_sg,
            "tumor_eps_gt": roi_full_3t["tumor_eps_gt"],
            "healthy_sar": hsar_cg_sg,
            "tumor_sar": tsar_cg_sg,
        },
        {
            "method": "CG-SENSE + cr-MREPT (Convección-Reacción)",
            "b0_tesla": 3.0,
            "zf_psnr": zf_psnr_3t,
            "cg_psnr": cg_psnr_3t,
            "psnr_db": cg_psnr_3t,
            "ssim": cg_ssim_3t,
            "nrmse_sigma": nrmse_sig_cg_cr,
            "nrmse_eps": nrmse_eps_cg_sg,
            "tumor_sigma": tsig_cg_cr,
            "tumor_sigma_gt": roi_full_3t["tumor_sigma_gt"],
            "tumor_eps": teps_cg_sg,
            "tumor_eps_gt": roi_full_3t["tumor_eps_gt"],
            "healthy_sar": hsar_cg_cr,
            "tumor_sar": tsar_cg_cr,
        },
        {
            "method": "Ablación 1: Imagen PINN-SENSE + SG-EPT Post-hoc",
            "b0_tesla": 3.0,
            "zf_psnr": zf_psnr_3t,
            "cg_psnr": cg_psnr_3t,
            "psnr_db": h_no_ept[-1].psnr_db,
            "ssim": h_no_ept[-1].ssim,
            "nrmse_sigma": nrmse_sig_pinn_sg,
            "nrmse_eps": nrmse_eps_pinn_sg,
            "tumor_sigma": tsig_pinn_sg,
            "tumor_sigma_gt": roi_full_3t["tumor_sigma_gt"],
            "tumor_eps": teps_pinn_sg,
            "tumor_eps_gt": roi_full_3t["tumor_eps_gt"],
            "healthy_sar": hsar_pinn_sg,
            "tumor_sar": tsar_pinn_sg,
        },
        {
            "method": "Ablación 2: PINN sin Desenvolvimiento $\\mathbb{S}^1$ ($2\\pi$)",
            "b0_tesla": 7.0,
            "zf_psnr": zf_psnr_7t,
            "cg_psnr": cg_psnr_7t,
            "psnr_db": h_no_s1_7t[-1].psnr_db,
            "ssim": h_no_s1_7t[-1].ssim,
            "nrmse_sigma": h_no_s1_7t[-1].nrmse_sigma,
            "nrmse_eps": h_no_s1_7t[-1].nrmse_eps,
            "tumor_sigma": roi_no_s1_7t["tumor_sigma_pred"],
            "tumor_sigma_gt": roi_no_s1_7t["tumor_sigma_gt"],
            "tumor_eps": roi_no_s1_7t["tumor_eps_pred"],
            "tumor_eps_gt": roi_no_s1_7t["tumor_eps_gt"],
            "healthy_sar": roi_no_s1_7t["healthy_sar_pred"],
            "tumor_sar": roi_no_s1_7t["tumor_sar_pred"],
        },
        {
            "method": "\\textbf{Propuesta: Decoupled MRI-PINN Completa (3.0 T)}",
            "b0_tesla": 3.0,
            "zf_psnr": zf_psnr_3t,
            "cg_psnr": cg_psnr_3t,
            "psnr_db": h_full_3t[-1].psnr_db,
            "ssim": h_full_3t[-1].ssim,
            "nrmse_sigma": h_full_3t[-1].nrmse_sigma,
            "nrmse_eps": h_full_3t[-1].nrmse_eps,
            "tumor_sigma": roi_full_3t["tumor_sigma_pred"],
            "tumor_sigma_gt": roi_full_3t["tumor_sigma_gt"],
            "tumor_eps": roi_full_3t["tumor_eps_pred"],
            "tumor_eps_gt": roi_full_3t["tumor_eps_gt"],
            "healthy_sar": roi_full_3t["healthy_sar_pred"],
            "tumor_sar": roi_full_3t["tumor_sar_pred"],
        },
        {
            "method": "\\textbf{Propuesta: Decoupled MRI-PINN Completa (7.0 T)}",
            "b0_tesla": 7.0,
            "zf_psnr": zf_psnr_7t,
            "cg_psnr": cg_psnr_7t,
            "psnr_db": h_full_7t[-1].psnr_db,
            "ssim": h_full_7t[-1].ssim,
            "nrmse_sigma": h_full_7t[-1].nrmse_sigma,
            "nrmse_eps": h_full_7t[-1].nrmse_eps,
            "tumor_sigma": roi_full_7t["tumor_sigma_pred"],
            "tumor_sigma_gt": roi_full_7t["tumor_sigma_gt"],
            "tumor_eps": roi_full_7t["tumor_eps_pred"],
            "tumor_eps_gt": roi_full_7t["tumor_eps_gt"],
            "healthy_sar": roi_full_7t["healthy_sar_pred"],
            "tumor_sar": roi_full_7t["tumor_sar_pred"],
        },
    ]
    for r in results["ablation_and_baselines"]:
        print(
            f"  * {r['method']:<50} | B0={r['b0_tesla']}T | ZF: {r['zf_psnr']:.2f} | CG: {r['cg_psnr']:.2f} -> PSNR: {r['psnr_db']:.2f} dB | "
            f"SSIM: {r['ssim']:.4f} | NRMSE_sig: {r['nrmse_sigma']:.4f} | NRMSE_eps: {r['nrmse_eps']:.4f} | "
            f"ROI sig: {r['tumor_sigma']:.2f}/{r['tumor_sigma_gt']:.2f} S/m | ROI eps: {r['tumor_eps']:.1f}/{r['tumor_eps_gt']:.1f}"
        )

    # -------------------------------------------------------------------------
    # ESTUDIO 3: Trayectorias K-Space Emparejadas en Fraccion (R=4x @ 25%, R=8x @ 12.5%)
    # -------------------------------------------------------------------------
    print("\n[Estudio 3/5] Evaluando Trayectorias K-Space Emparejadas (Cartesian, Radial, Poisson) a R=4x y R=8x...")
    stress_8x_maps: Dict[str, Dict[str, Any]] = {}
    for acc in [4, 8]:
        for mtype in ["cartesian", "radial", "poisson"]:
            if acc == 4 and mtype == "poisson":
                ds = ds_3t
                last = h_full_3t[-1]
                maps = m_full_3t
                zf_psnr = zf_psnr_3t
                zf_ssim = zf_ssim_3t
                cg_psnr = cg_psnr_3t
                roi = roi_full_3t
            else:
                torch.manual_seed(108)
                np.random.seed(108)
                ds = generate_clinical_in_vivo_brain_phantom(
                    b0_tesla=3.0,
                    noise_snr_db=33.0,
                    mask_type=mtype,
                    acceleration=acc,
                    acs_lines=18,
                    seed=108,
                    device=device,
                )
                torch.manual_seed(108)
                np.random.seed(108)
                cfg = _base_config(b0_tesla=3.0, mask_type=mtype, acceleration=acc, epochs=epochs)
                trainer = MRIPINNTrainer(dataset=ds, config=cfg, device=device)
                history, maps = trainer.train(epochs=epochs, verbose=False)
                last = history[-1]
                zf_psnr = compute_psnr(torch.abs(ds.zero_filled), ds.m_mag_gt)
                zf_ssim = compute_ssim(torch.abs(ds.zero_filled), ds.m_mag_gt)
                cg_cplx = compute_cg_sense(ds.kspace_masked, trainer.coil_acs_ref, ds.mask)
                cg_psnr = compute_psnr(torch.abs(cg_cplx), ds.m_mag_gt)
                roi = _extract_roi_metrics(ds, maps)
            samp_pct = float(torch.mean(ds.mask).item()) * 100.0

            entry = {
                "acceleration": acc,
                "mask_type": mtype,
                "sampling_pct": samp_pct,
                "zf_psnr": zf_psnr,
                "zf_ssim": zf_ssim,
                "cg_psnr": cg_psnr,
                "pinn_psnr": last.psnr_db,
                "pinn_ssim": last.ssim,
                "nrmse_sigma": last.nrmse_sigma,
                "nrmse_eps": last.nrmse_eps,
                **roi,
            }
            results["trajectories_stress"].append(entry)
            if acc == 8:
                stress_8x_maps[mtype] = {
                    "mask": ds.mask.detach().cpu().numpy(),
                    "zf_mag": torch.abs(ds.zero_filled).detach().cpu().numpy(),
                    "m_mag": maps["m_mag"].detach().cpu().numpy(),
                    "sigma": maps["sigma"].detach().cpu().numpy(),
                    "sigma_err": torch.abs((maps["sigma"] - ds.sigma_gt) * ds.tissue_mask_gt).detach().cpu().numpy(),
                    "entry": entry,
                }
            print(
                f"  * R={acc}x {mtype.upper():9s} ({samp_pct:4.1f}% k-space) | "
                f"ZF: {zf_psnr:.2f} | CG: {cg_psnr:.2f} -> PINN: {last.psnr_db:.2f} dB (+{last.psnr_db - zf_psnr:.2f} dB) | "
                f"SSIM: {last.ssim:.4f} | NRMSE_sig: {last.nrmse_sigma:.4f} | NRMSE_eps: {last.nrmse_eps:.4f} | ROI sig: {roi['tumor_sigma_pred']:.2f}/{roi['tumor_sigma_gt']:.2f} S/m"
            )

    # -------------------------------------------------------------------------
    # ESTUDIO 4: Control de Confusion Magnitud vs Conductividad (Major #2)
    # -------------------------------------------------------------------------
    print("\n[Estudio 4/5] Ejecutando Control de Confusion Magnitud vs Conductividad (@ 3.0T, R=4x Poisson)...")
    for cmode, clabel in [
        ("coupled", "Acoplado Estándar ($|m|_{\\text{ROI}}=1.00, \\sigma_{\\text{GT}}=1.20$)"),
        ("sigma_only", "Control A: Isointenso en $|m|$ ($0.61$), Hiper-conductivo ($\\sigma_{\\text{GT}}=1.20$)"),
        ("mag_only", "Control B: Hiperintenso en $|m|$ ($1.00$), Isoconductivo ($\\sigma_{\\text{GT}}=0.34$)"),
    ]:
        if cmode == "coupled":
            ds_c = ds_3t
            last_c = h_full_3t[-1]
            zf_c = zf_psnr_3t
            cg_c = cg_psnr_3t
            roi_c = roi_full_3t
        else:
            torch.manual_seed(108)
            np.random.seed(108)
            ds_c = generate_clinical_in_vivo_brain_phantom(
                b0_tesla=3.0,
                noise_snr_db=33.0,
                mask_type="poisson",
                acceleration=4,
                seed=108,
                contrast_mode=cmode,
                device=device,
            )
            torch.manual_seed(108)
            np.random.seed(108)
            cfg_c = _base_config(b0_tesla=3.0, mask_type="poisson", acceleration=4, epochs=epochs)
            tr_c = MRIPINNTrainer(dataset=ds_c, config=cfg_c, device=device)
            h_c, m_c = tr_c.train(epochs=epochs, verbose=False)
            last_c = h_c[-1]
            zf_c = compute_psnr(torch.abs(ds_c.zero_filled), ds_c.m_mag_gt)
            cg_cplx_c = compute_cg_sense(ds_c.kspace_masked, tr_c.coil_acs_ref, ds_c.mask)
            cg_c = compute_psnr(torch.abs(cg_cplx_c), ds_c.m_mag_gt)
            roi_c = _extract_roi_metrics(ds_c, m_c)

        c_entry = {
            "contrast_mode": cmode,
            "label": clabel,
            "zf_psnr": zf_c,
            "cg_psnr": cg_c,
            "psnr_db": last_c.psnr_db,
            "ssim": last_c.ssim,
            "nrmse_sigma": last_c.nrmse_sigma,
            "nrmse_eps": last_c.nrmse_eps,
            "tumor_sigma_gt": roi_c["tumor_sigma_gt"],
            "tumor_sigma_pred": roi_c["tumor_sigma_pred"],
            "healthy_sigma_pred": roi_c["healthy_sigma_pred"],
        }
        results["confounding_controls"].append(c_entry)
        print(
            f"  * {cmode:<12} | ZF: {zf_c:.2f} | CG: {cg_c:.2f} -> PINN: {c_entry['psnr_db']:.2f} dB | "
            f"SSIM: {c_entry['ssim']:.4f} | NRMSE_sig: {c_entry['nrmse_sigma']:.4f} | NRMSE_eps: {c_entry['nrmse_eps']:.4f} | "
            f"GT ROI: {c_entry['tumor_sigma_gt']:.2f} S/m -> Pred ROI: {c_entry['tumor_sigma_pred']:.2f} S/m "
            f"(WM sano: {c_entry['healthy_sigma_pred']:.2f} S/m)"
        )

    # -------------------------------------------------------------------------
    # ESTUDIO 5: Barrido de Sensibilidad a SNR en Espacio-k (20, 26, 33, 40 dB)
    # -------------------------------------------------------------------------
    print("\n[Estudio 5/5] Evaluando Sensibilidad al Ruido Termico SNR (20, 26, 33, 40 dB @ 3.0T)...")
    for snr_db in [20.0, 26.0, 33.0, 40.0]:
        if abs(snr_db - 33.0) < 1e-3:
            ds_snr = ds_3t
            last_snr = h_full_3t[-1]
            zf_snr = zf_psnr_3t
            cg_snr = cg_psnr_3t
            roi_snr = roi_full_3t
        else:
            torch.manual_seed(108)
            np.random.seed(108)
            ds_snr = generate_clinical_in_vivo_brain_phantom(
                b0_tesla=3.0,
                noise_snr_db=snr_db,
                mask_type="poisson",
                acceleration=4,
                seed=108,
                device=device,
            )
            torch.manual_seed(108)
            np.random.seed(108)
            cfg_snr = _base_config(b0_tesla=3.0, mask_type="poisson", acceleration=4, epochs=epochs)
            tr_snr = MRIPINNTrainer(dataset=ds_snr, config=cfg_snr, device=device)
            h_snr, m_snr = tr_snr.train(epochs=epochs, verbose=False)
            last_snr = h_snr[-1]
            zf_snr = compute_psnr(torch.abs(ds_snr.zero_filled), ds_snr.m_mag_gt)
            cg_cplx_snr = compute_cg_sense(ds_snr.kspace_masked, tr_snr.coil_acs_ref, ds_snr.mask)
            cg_snr = compute_psnr(torch.abs(cg_cplx_snr), ds_snr.m_mag_gt)
            roi_snr = _extract_roi_metrics(ds_snr, m_snr)
        results["snr_sensitivity"].append(
            {
                "snr_db": snr_db,
                "zf_psnr": zf_snr,
                "cg_psnr": cg_snr,
                "psnr_db": last_snr.psnr_db,
                "ssim": last_snr.ssim,
                "nrmse_sigma": last_snr.nrmse_sigma,
                "nrmse_eps": last_snr.nrmse_eps,
                "tumor_sigma_pred": roi_snr["tumor_sigma_pred"],
                "tumor_sigma_gt": roi_snr["tumor_sigma_gt"],
            }
        )
        print(
            f"  * SNR={snr_db:4.1f} dB | ZF: {zf_snr:.2f} | CG: {cg_snr:.2f} -> PINN: {last_snr.psnr_db:.2f} dB | "
            f"SSIM: {last_snr.ssim:.4f} | NRMSE_sig: {last_snr.nrmse_sigma:.4f} | "
            f"NRMSE_eps: {last_snr.nrmse_eps:.4f} | ROI sig: {roi_snr['tumor_sigma_pred']:.2f}/{roi_snr['tumor_sigma_gt']:.2f} S/m"
        )

    # Guardar JSON de telemetria preservando external_blind_validation si existe
    json_path = out_dir / "benchmark_metrics.json"
    if json_path.exists():
        try:
            old_json = json.loads(json_path.read_text(encoding="utf-8"))
            if "external_blind_validation" in old_json:
                results["external_blind_validation"] = old_json["external_blind_validation"]
        except Exception:
            pass
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # CONFIGURACION DE ESTILO EDITORIAL PARA IMPRESION (FONDO BLANCO RACCEFYN)
    # -------------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#0f172a",
            "axes.labelcolor": "#1e293b",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )

    # -------------------------------------------------------------------------
    # GENERAR FIGURA 1 (300 DPI, FONDO BLANCO): Multi-Campo + eps_r + Error |sigma - sigma_GT| + SAR
    # -------------------------------------------------------------------------
    print("\n[Exportando Figuras 300 DPI en Fondo Blanco] Generando fig1_multifield_ept_sar_300dpi.png...")
    fig1, axes1 = plt.subplots(3, 5, figsize=(16.5, 9.8), dpi=300)

    for row_idx, b0 in enumerate([1.5, 3.0, 7.0]):
        mdata = multifield_maps[b0]
        ent = results["multifield_numerical"][row_idx]
        sar_max = 8.0 if b0 >= 6.5 else (2.5 if b0 >= 2.5 else 0.8)

        panels_row = [
            (mdata["m_mag"], "gray", 0.0, 1.0, f"|m(x)| @ {b0} T ({ent['pinn_psnr']:.1f} dB)", "u.a."),
            (mdata["phi_tr"], "twilight", float(mdata["phi_tr"].min()), float(mdata["phi_tr"].max()), f"Fase φ_tr(x) ({ent['larmor_mhz']} MHz)", "rad"),
            (mdata["sigma"], "viridis", 0.01, 1.80, f"σ(x) (NRMSE {ent['nrmse_sigma']:.3f})", "S/m"),
            (mdata["eps_r"], "plasma", 1.0, 88.0, f"ε_r(x) (NRMSE {ent['nrmse_eps']:.3f})", "adim."),
            (mdata["sar"], "inferno", 0.0, sar_max, f"SAR(x) (ROI {ent['tumor_sar_pred']:.2f} W/kg)", "W/kg"),
        ]

        for col_idx, (arr, cmap, vmin, vmax, title, unit) in enumerate(panels_row):
            ax = axes1[row_idx, col_idx]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, color="#0f172a", fontsize=8.6, fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#94a3b8")
                spine.set_linewidth(0.7)

            if col_idx in (0, 2, 3, 4):
                ell = Ellipse(
                    xy=(86.5, 53.5),
                    width=16.5,
                    height=15.5,
                    edgecolor="#00e5ff" if col_idx == 4 else "#e11d48",
                    facecolor="none",
                    linestyle="--",
                    linewidth=1.3,
                )
                ax.add_patch(ell)

            if row_idx == 0 and col_idx == 0:
                _add_scale_bar_5cm(ax)

            cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7.2, colors="#1e293b")
            cbar.set_label(unit, color="#334155", fontsize=7.2)

    fig1.tight_layout(pad=0.9)
    fig1_path = out_dir / "fig1_multifield_ept_sar_300dpi.png"
    fig1.savefig(fig1_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # GENERAR FIGURA 2 (300 DPI, FONDO BLANCO): Trayectorias R=8x (12.5%) + Mapa de Error + Perfil 1D
    # -------------------------------------------------------------------------
    print("[Exportando Figuras 300 DPI en Fondo Blanco] Generando fig2_ablation_8x_trajectories_300dpi.png...")
    fig2, axes2 = plt.subplots(3, 5, figsize=(16.8, 9.8), dpi=300)
    sigma_gt_3t = cortical_ds_base.sigma_gt.detach().cpu().numpy()
    tumor_row_idx = 54  # Fila axial que atraviesa el centro del glioma temporoparietal (cy = -0.16)
    x_mm = np.linspace(-120.0, 120.0, 128)

    for row_idx, mtype in enumerate(["cartesian", "radial", "poisson"]):
        sdata = stress_8x_maps[mtype]
        ent = sdata["entry"]
        m_label = {"cartesian": "Cartesiano 1D", "radial": "Radial Áureo", "poisson": "Poisson-disc 2D"}[mtype]

        row_specs = [
            (sdata["mask"], "gray", 0.0, 1.0, f"{m_label} R=8x ({ent['sampling_pct']:.1f}%)", "P(k)"),
            (sdata["zf_mag"], "gray", 0.0, 1.0, f"Adjunto ZF ({ent['zf_psnr']:.2f} dB)", "u.a."),
            (sdata["m_mag"], "gray", 0.0, 1.0, f"PINN |m(x)| ({ent['pinn_psnr']:.2f} dB)", "u.a."),
            (sdata["sigma"], "viridis", 0.01, 1.70, f"EPT σ(x) (ROI {ent['tumor_sigma_pred']:.2f} S/m)", "S/m"),
            (sdata["sigma_err"], "magma", 0.0, 0.45, f"Error |σ - σ_GT| (NRMSE {ent['nrmse_sigma']:.3f})", "S/m"),
        ]

        for col_idx, (arr, cmap, vmin, vmax, title, unit) in enumerate(row_specs):
            ax = axes2[row_idx, col_idx]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, color="#0f172a", fontsize=8.5, fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#94a3b8")
                spine.set_linewidth(0.7)

            if col_idx in (2, 3, 4):
                ell = Ellipse(
                    xy=(86.5, 53.5),
                    width=16.5,
                    height=15.5,
                    edgecolor="#e11d48" if col_idx != 4 else "#38bdf8",
                    facecolor="none",
                    linestyle="--",
                    linewidth=1.3,
                )
                ax.add_patch(ell)

            if row_idx == 0 and col_idx == 2:
                _add_scale_bar_5cm(ax)

            cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7.2, colors="#1e293b")
            cbar.set_label(unit, color="#334155", fontsize=7.2)

    fig2.tight_layout(pad=0.9)
    fig2_path = out_dir / "fig2_ablation_8x_trajectories_300dpi.png"
    fig2.savefig(fig2_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig2)

    elapsed_s = time.perf_counter() - t_start
    print("=" * 96)
    print(f" SUITE COMPLETADA EN {elapsed_s:.1f} s | ARTEFACTOS EXPORTADOS EN: {out_dir.resolve()}")
    print("=" * 96)


if __name__ == "__main__":
    main()
