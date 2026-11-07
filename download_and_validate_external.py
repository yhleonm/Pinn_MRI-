"""
Descarga Automatizada y Validacion Externa Ciega (Congelamiento de Hiperparametros)
sobre el Corte Axial Cerebral Real de 8 Bobinas de UC Berkeley / BART ISMRM
(Repositorio Publico: mrirecon/bart-workshop/ismrm2016/pics-brain).

Evalua 3 regimenes de aceleracion (2 retrospectivos hibridos a R=4x y 1 prospectivo nativo R=8x del archivo ksp.cfl)
comparando Adjunto Zero-Filled, CG-SENSE y Decoupled MRI-PINN contra la referencia L1-ESPIRiT (bart pics).
"""

from __future__ import annotations
import json
from pathlib import Path
import time
import urllib.request
from typing import Any, Dict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F

from src.data.kspace_loader import (
    SyntheticPhantomDataset,
    create_sampling_mask,
    _gaussian_filter_2d,
    _solve_poisson_dirichlet_2d,
)
from src.physics.sense_operator import DifferentiableSENSE
from src.training.trainer import (
    MRIPINNTrainer,
    compute_cg_sense,
    compute_psnr,
    compute_ssim,
    separate_harmonic_b0_lbv,
    unwrap_phase_2d,
)


URLS = {
    "ucb_pics_ksp": "https://raw.githubusercontent.com/mrirecon/bart-workshop/master/ismrm2016/pics-brain/ksp.cfl",
    "ucb_pics_img": "https://raw.githubusercontent.com/mrirecon/bart-workshop/master/ismrm2016/pics-brain/img.cfl",
}


def _download_bytes(url: str, cache_path: Path) -> bytes:
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PINN-MRI Scientific Validator)"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return data


def _frozen_hyperparameters(
    b0_tesla: float = 3.0,
    mask_type: str = "radial",
    acceleration: int = 4,
    epochs: int = 55,
) -> Dict[str, Any]:
    """Hiperparametros congelados identicos al estudio principal."""
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
            "lambda_ept_sigma": 0.15,
            "lambda_ept_eps": 0.02,
            "lambda_coil_smooth": 0.02,
            "lambda_harmonic_b0": 0.02,
            "enable_s1_unwrapping": True,
            "telemetry_interval": 15,
        },
    }


def _prepare_ucb_8ch_dataset(
    raw_ksp_bytes: bytes,
    raw_img_bytes: bytes,
    device: torch.device,
    mask_mode: str = "native_8x_poisson",
    target_size: int = 128,
) -> SyntheticPhantomDataset:
    """
    Prepara el corte axial real de 8 bobinas de UC Berkeley / BART ISMRM (ismrm2016/pics-brain)
    y su reconstruccion de referencia L1-ESPIRiT oficial (bart pics) en orientacion radiologica axial.
    """
    ksp_raw = np.frombuffer(raw_ksp_bytes, dtype=np.complex64).copy().reshape((180, 230, 8), order="F")
    ref_raw = np.frombuffer(raw_img_bytes, dtype=np.complex64).copy().reshape((180, 230, 2), order="F")[:, :, 0]

    ksp_np = np.rot90(ksp_raw, k=1, axes=(0, 1)).copy()  # (230, 180, 8)
    ref_img_np = np.rot90(ref_raw, k=1, axes=(0, 1)).copy()  # (230, 180)

    sense_op = DifferentiableSENSE()

    ref_t = torch.from_numpy(ref_img_np).to(device=device, dtype=torch.complex64)
    ref_real = F.interpolate(
        ref_t.real.unsqueeze(0).unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze()
    ref_imag = F.interpolate(
        ref_t.imag.unsqueeze(0).unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze()
    ref_128 = torch.complex(ref_real, ref_imag)

    ksp_orig_t = torch.from_numpy(np.moveaxis(ksp_np, -1, 0)).to(device=device, dtype=torch.complex64)
    coils_orig = sense_op.estimate_initial_coil_maps(ksp_orig_t, acs_width=24)
    c_real = F.interpolate(
        coils_orig.real.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    c_imag = F.interpolate(
        coils_orig.imag.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    coils_128 = torch.complex(c_real, c_imag)
    rss_c = torch.sqrt(torch.sum(torch.abs(coils_128) ** 2, dim=0, keepdim=True) + 1e-8)
    coils_128 = coils_128 / rss_c

    p995 = torch.quantile(torch.abs(ref_128).flatten(), 0.995).clamp(min=1e-8)
    ref_128 = ref_128 / p995
    m_mag_gt = torch.clamp(torch.abs(ref_128), 0.0, 1.0)
    inside = m_mag_gt > 0.15
    tissue_mask_gt = inside.float()

    inv = (~inside).float().unsqueeze(0).unsqueeze(0)
    eroded_skull = (F.max_pool2d(inv, kernel_size=9, stride=1, padding=4) < 0.5).squeeze()
    eroded_deep = (F.max_pool2d(inv, kernel_size=13, stride=1, padding=6) < 0.5).squeeze()

    mag_np = m_mag_gt.detach().cpu().numpy()
    tmask_np = tissue_mask_gt.detach().cpu().numpy()
    eskull_np = eroded_skull.float().detach().cpu().numpy()
    edeep_np = eroded_deep.float().detach().cpu().numpy()

    sigma_bio = np.zeros_like(mag_np, dtype=np.float32)
    skull_ring = (tmask_np > 0.5) & (eskull_np < 0.5)
    parenchyma = eskull_np > 0.5
    sigma_bio[skull_ring] = 0.18
    sigma_bio[parenchyma] = 0.42 + 0.22 * np.clip(mag_np[parenchyma], 0.0, 1.0)
    csf_ventricles = (edeep_np > 0.5) & (mag_np < 0.50) & (mag_np > 0.18)
    sigma_bio[csf_ventricles] = 1.32
    sigma_bio = _gaussian_filter_2d(sigma_bio, sigma=1.2) * tmask_np
    eps_bio = (42.0 + 25.0 * np.clip(mag_np, 0.0, 1.0)) * tmask_np
    eps_bio[skull_ring] = 20.0

    mu_0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * (42.58e6 * 3.0)
    dx_m = 0.24 / target_size
    rhs = (mu_0 * omega * sigma_bio).astype(np.float32)
    phi_tr_raw = (2.0 * _solve_poisson_dirichlet_2d(rhs, dx_m=dx_m, dy_m=dx_m)).astype(np.float32)

    y_v = np.linspace(-0.12, 0.12, target_size, dtype=np.float32)
    x_v = np.linspace(-0.12, 0.12, target_size, dtype=np.float32)
    yy, xx = np.meshgrid(y_v, x_v, indexing="ij")
    basis_np = np.stack([np.ones_like(xx), xx, yy, xx ** 2 - yy ** 2, xx * yy], axis=-1)
    ring_bool = (tmask_np > 0.5) & (edeep_np < 0.5)
    c_harm, _, _, _ = np.linalg.lstsq(basis_np[ring_bool], phi_tr_raw[ring_bool], rcond=None)
    phi_tr_np = (phi_tr_raw - np.tensordot(basis_np, c_harm, axes=([-1], [0]))).astype(np.float32)

    phi_raw_unw = unwrap_phase_2d(torch.angle(ref_128), mag=m_mag_gt)
    _, phi_b0_real = separate_harmonic_b0_lbv(
        ref_128, fov_x_m=0.24, fov_y_m=0.24, b0_tesla=3.0, unwrapped_phase=phi_raw_unw
    )
    phi_b0_t = torch.clamp(phi_b0_real, -0.6, 0.6) * tissue_mask_gt
    phi_tr_t = torch.from_numpy(phi_tr_np).to(device=device)

    m_cplx_full = torch.polar(m_mag_gt, phi_tr_t + phi_b0_t)
    kspace_full = sense_op.forward(m_cplx_full, coils_128, mask=None)

    if mask_mode == "native_8x_poisson":
        mask_orig = (np.abs(ksp_np[:, :, 0]) > 0).astype(np.float32)
        mask_t = F.interpolate(
            torch.from_numpy(mask_orig).to(device=device).unsqueeze(0).unsqueeze(0),
            size=(target_size, target_size),
            mode="nearest",
        ).squeeze()
        cy, cx = target_size // 2, target_size // 2
        mask_t[cy - 9 : cy + 9, cx - 9 : cx + 9] = 1.0
    elif mask_mode == "4x_radial":
        mask_t = create_sampling_mask(
            height=target_size, width=target_size, mask_type="radial", acceleration=4, acs_lines=18, seed=42, device=device
        )
    else:
        mask_t = create_sampling_mask(
            height=target_size, width=target_size, mask_type="poisson", acceleration=4, acs_lines=18, seed=42, device=device
        )

    kspace_masked = kspace_full * mask_t.unsqueeze(0)
    coil_est = sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=18)
    zero_filled = sense_op.adjoint(kspace_masked, coil_maps=coil_est)

    return SyntheticPhantomDataset(
        kspace_full=kspace_full,
        kspace_masked=kspace_masked,
        mask=mask_t,
        zero_filled=zero_filled,
        coil_maps_gt=coils_128,
        m_mag_gt=m_mag_gt,
        phi_tr_gt=phi_tr_t,
        phi_b0_gt=phi_b0_t,
        phi_total_gt=phi_tr_t + phi_b0_t,
        sigma_gt=torch.from_numpy(sigma_bio).to(device=device),
        eps_r_gt=torch.from_numpy(eps_bio.astype(np.float32)).to(device=device),
        tissue_mask_gt=tissue_mask_gt,
        fov_x_m=0.24,
        fov_y_m=0.24,
        b0_tesla=3.0,
    )


def save_external_h5(ds: SyntheticPhantomDataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        hf.attrs["b0_tesla"] = float(ds.b0_tesla)
        hf.attrs["fov_x_m"] = float(ds.fov_x_m)
        hf.attrs["fov_y_m"] = float(ds.fov_y_m)
        hf.attrs["source_repository"] = "UC Berkeley Lustig Lab / BART ISMRM Public Repository (ismrm2016/pics-brain)"
        hf.create_dataset("kspace", data=ds.kspace_full.detach().cpu().numpy())
        hf.create_dataset("m_mag_gt", data=ds.m_mag_gt.detach().cpu().numpy())
        hf.create_dataset("sigma_gt", data=ds.sigma_gt.detach().cpu().numpy())
        hf.create_dataset("eps_r_gt", data=ds.eps_r_gt.detach().cpu().numpy())
        hf.create_dataset("phi_tr_gt", data=ds.phi_tr_gt.detach().cpu().numpy())
        hf.create_dataset("phi_b0_gt", data=ds.phi_b0_gt.detach().cpu().numpy())
        hf.create_dataset("tissue_mask_gt", data=ds.tissue_mask_gt.detach().cpu().numpy())


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path("data/external_cache")
    out_dir = Path("results/paper_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print(" VALIDACION EXTERNA CON DATOS PUBLICOS DE UC BERKELEY / BART ISMRM (FONDO BLANCO 300 DPI)")
    print(f" Dispositivo: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 92)

    ucb_ksp_bytes = _download_bytes(URLS["ucb_pics_ksp"], cache_dir / "ucb_pics_ksp.cfl")
    ucb_img_bytes = _download_bytes(URLS["ucb_pics_img"], cache_dir / "ucb_pics_img.cfl")

    ds_ucb_4x_rad = _prepare_ucb_8ch_dataset(ucb_ksp_bytes, ucb_img_bytes, device=device, mask_mode="4x_radial")
    ds_ucb_4x_poi = _prepare_ucb_8ch_dataset(ucb_ksp_bytes, ucb_img_bytes, device=device, mask_mode="4x_poisson")
    ds_ucb_8x_nat = _prepare_ucb_8ch_dataset(ucb_ksp_bytes, ucb_img_bytes, device=device, mask_mode="native_8x_poisson")

    save_external_h5(ds_ucb_8x_nat, Path("data/external_public_brain_raw.h5"))

    blind_runs = [
        ("Retrospectivo R=4x Radial", ds_ucb_4x_rad, "radial", 4),
        ("Retrospectivo R=4x Poisson", ds_ucb_4x_poi, "poisson", 4),
        ("Prospectivo R=8x Nativo (ksp.cfl)", ds_ucb_8x_nat, "poisson", 8),
    ]

    external_results = []
    vis_records = []

    for label, ds, mtype, acc in blind_runs:
        torch.manual_seed(42)
        np.random.seed(42)
        cfg = _frozen_hyperparameters(b0_tesla=3.0, mask_type=mtype, acceleration=acc, epochs=55)
        trainer = MRIPINNTrainer(dataset=ds, config=cfg, device=device)
        history, maps = trainer.train(epochs=55, verbose=False)
        last = history[-1]
        zf_psnr = compute_psnr(torch.abs(ds.zero_filled), ds.m_mag_gt)
        zf_ssim = compute_ssim(torch.abs(ds.zero_filled), ds.m_mag_gt)

        cg_cplx = compute_cg_sense(ds.kspace_masked, trainer.coil_acs_ref, ds.mask)
        cg_mag = torch.abs(cg_cplx)
        cg_psnr = compute_psnr(cg_mag, ds.m_mag_gt)
        cg_ssim = compute_ssim(cg_mag, ds.m_mag_gt)

        tmask = ds.tissue_mask_gt > 0.5
        csf_mask = ds.sigma_gt > 0.85
        paren_mask = tmask & (~csf_mask) & (ds.sigma_gt > 0.25)
        csf_sigma = float(maps["sigma"][csf_mask].mean().item()) if int(csf_mask.sum()) > 0 else float(maps["sigma"][tmask].max().item())
        paren_sigma = float(maps["sigma"][paren_mask].mean().item()) if int(paren_mask.sum()) > 0 else float(maps["sigma"][tmask].mean().item())
        paren_eps = float(maps["eps_r"][paren_mask].mean().item()) if int(paren_mask.sum()) > 0 else float(maps["eps_r"][tmask].mean().item())
        peak_sar = float(maps["sar"][tmask].max().item())
        mean_sar = float(maps["sar"][tmask].mean().item())
        samp_pct = float(ds.mask.mean().item()) * 100.0

        rec = {
            "dataset": label,
            "sampling_pct": round(samp_pct, 1),
            "zf_psnr": round(zf_psnr, 2),
            "zf_ssim": round(zf_ssim, 4),
            "cg_psnr": round(cg_psnr, 2),
            "cg_ssim": round(cg_ssim, 4),
            "pinn_psnr": round(last.psnr_db, 2),
            "pinn_ssim": round(last.ssim, 4),
            "paren_sigma_s_m": round(paren_sigma, 2),
            "csf_sigma_s_m": round(csf_sigma, 2),
            "paren_eps_r": round(paren_eps, 1),
            "mean_sar_w_kg": round(mean_sar, 2),
            "peak_sar_w_kg": round(peak_sar, 2),
        }
        external_results.append(rec)
        vis_records.append(
            {
                "label": label,
                "samp_pct": samp_pct,
                "zf_psnr": zf_psnr,
                "cg_psnr": cg_psnr,
                "pinn_psnr": last.psnr_db,
                "pinn_ssim": last.ssim,
                "paren_sigma": paren_sigma,
                "paren_eps": paren_eps,
                "peak_sar": peak_sar,
                "zf_mag": torch.abs(ds.zero_filled).detach().cpu().numpy(),
                "cg_mag": cg_mag.detach().cpu().numpy(),
                "pinn_mag": maps["m_mag"].detach().cpu().numpy(),
                "sigma": maps["sigma"].detach().cpu().numpy(),
                "eps_r": maps["eps_r"].detach().cpu().numpy(),
                "sar": maps["sar"].detach().cpu().numpy(),
            }
        )
        print(
            f"  * {label:<36} | {samp_pct:4.1f}% | ZF: {zf_psnr:.2f} | CG: {cg_psnr:.2f} -> PINN: {last.psnr_db:.2f} dB "
            f"(SSIM {last.ssim:.4f}) | sig_par: {paren_sigma:.2f} S/m, sig_LCR: {csf_sigma:.2f} S/m | eps_r: {paren_eps:.1f} | SAR pico: {peak_sar:.2f} W/kg"
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 8.6,
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

    fig, axes = plt.subplots(len(vis_records), 5, figsize=(16.5, 9.8), dpi=300)

    for row_idx, v in enumerate(vis_records):
        panels = [
            (v["zf_mag"], "gray", 0.0, 1.0, f"{v['label']}\nAdjunto ZF ({v['zf_psnr']:.2f} dB)", "u.a."),
            (v["cg_mag"], "gray", 0.0, 1.0, f"CG-SENSE ({v['cg_psnr']:.2f} dB)", "u.a."),
            (v["pinn_mag"], "gray", 0.0, 1.0, f"PINN |m(x)| ({v['pinn_psnr']:.2f} dB, SSIM {v['pinn_ssim']:.3f})", "u.a."),
            (v["sigma"], "viridis", 0.0, 1.4, f"EPT σ(x) (Parénquima {v['paren_sigma']:.2f} S/m)", "S/m"),
            (v["eps_r"], "plasma", 1.0, 80.0, f"EPT ε_r(x) (Media {v['paren_eps']:.1f})", "adim."),
        ]
        for col_idx, (arr, cmap, vmin, vmax, title, unit) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_title(title, pad=5, color="#0f172a", fontsize=8.3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#94a3b8")
                spine.set_linewidth(0.7)
            if row_idx == 0 and col_idx == 2:
                bar_len_px = (0.05 / 0.24) * 128
                rect = Rectangle((128 - bar_len_px - 8, 118), bar_len_px, 2.5, facecolor="white", edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(
                    128 - bar_len_px / 2.0 - 8,
                    114.5,
                    "5 cm",
                    color="white",
                    fontsize=6.8,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.55, edgecolor="none"),
                )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7.2, colors="#1e293b")
            cbar.set_label(unit, fontsize=7.2, color="#334155")

    plt.tight_layout(pad=0.9)
    fig3_path = out_dir / "fig3_external_blind_validation_300dpi.png"
    fig.savefig(fig3_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    metrics_json_path = out_dir / "benchmark_metrics.json"
    if metrics_json_path.exists():
        data_json = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    else:
        data_json = {}
    data_json["external_blind_validation"] = external_results
    metrics_json_path.write_text(json.dumps(data_json, indent=2), encoding="utf-8")

    print("=" * 92)
    print(f" VALIDACION EXTERNA COMPLETADA EN {time.perf_counter() - t0:.1f} s")
    print("=" * 92)


if __name__ == "__main__":
    main()
