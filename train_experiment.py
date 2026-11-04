"""Main CLI script for PINN-MRI K-Space SENSE + Complex Helmholtz EPT experiments."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict
import yaml
import torch

from src.data.kspace_loader import (
    generate_shepp_logan_ept_phantom,
    load_h5_kspace_slice,
    save_dataset_to_h5,
)
from src.training.trainer import MRIPINNTrainer, compute_psnr, compute_ssim


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento y Validacion de DecoupledMRIPINN (SENSE + Helmholtz EPT)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Ruta al archivo YAML de configuracion.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "real_h5"],
        default="synthetic",
        help="Modo de ejecucion: 'synthetic' (Shepp-Logan 8 bobinas) o 'real_h5' (fastMRI/Calgary).",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/raw/sample_brain_8coils.h5",
        help="Ruta al archivo .h5 para el modo real_h5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Numero de epocas de entrenamiento.",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=None,
        help="Factor de aceleracion R (ej. 4 o 8).",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default=None,
        choices=["cartesian", "radial", "poisson"],
        help="Tipo de mascara de sub-muestreo en k-space.",
    )
    parser.add_argument(
        "--b0",
        type=float,
        default=None,
        help="Campo magnetico estatico B0 en Teslas (ej. 1.5 o 3.0).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    phys_cfg = cfg.setdefault("physics", {})
    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.acceleration is not None:
        data_cfg["acceleration"] = args.acceleration
    if args.mask_type is not None:
        data_cfg["mask_type"] = args.mask_type
    if args.b0 is not None:
        phys_cfg["b0_tesla"] = args.b0
        phys_cfg["larmor_freq_mhz"] = round(42.58 * args.b0, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(train_cfg.get("epochs", 50))
    matrix_size = data_cfg.get("matrix_size", [128, 128])
    h, w = int(matrix_size[0]), int(matrix_size[1])
    num_coils = int(data_cfg.get("num_coils", 8))
    mask_type = str(data_cfg.get("mask_type", "cartesian"))
    acceleration = int(data_cfg.get("acceleration", 4))
    acs_lines = int(data_cfg.get("acs_lines", 18))
    b0_tesla = float(phys_cfg.get("b0_tesla", 3.0))
    fov_x_m = float(phys_cfg.get("fov_x_m", 0.24))
    fov_y_m = float(phys_cfg.get("fov_y_m", 0.24))
    noise_snr_db = float(data_cfg.get("noise_snr_db", 35.0))

    print("=" * 88)
    print(" PINN-MRI K-SPACE + COMPLEX HELMHOLTZ EPT | SCIENTIFIC VERIFICATION")
    print("=" * 88)
    print(
        f" Mode: {args.mode} | Matrix: {h}x{w} | Coils: {num_coils} | "
        f"Mask: {mask_type} (R={acceleration}x, ACS={acs_lines}) | B0: {b0_tesla}T ({42.58 * b0_tesla:.2f} MHz)"
    )

    if args.mode == "synthetic":
        dataset = generate_shepp_logan_ept_phantom(
            height=h,
            width=w,
            num_coils=num_coils,
            fov_x_m=fov_x_m,
            fov_y_m=fov_y_m,
            b0_tesla=b0_tesla,
            noise_snr_db=noise_snr_db,
            mask_type=mask_type,
            acceleration=acceleration,
            acs_lines=acs_lines,
            device=device,
        )
        # Guardar copia .h5 de referencia en data/raw/ para pruebas del cargador .h5 y backend
        sample_h5 = Path("data/raw/sample_brain_8coils.h5")
        if not sample_h5.exists():
            save_dataset_to_h5(dataset, sample_h5)
    else:
        h5_path = Path(args.h5_path)
        if not h5_path.exists():
            print(f"[Info] No existe {h5_path}; generando archivo .h5 multi-bobina de prueba...")
            synth = generate_shepp_logan_ept_phantom(
                height=h,
                width=w,
                num_coils=num_coils,
                fov_x_m=fov_x_m,
                fov_y_m=fov_y_m,
                b0_tesla=b0_tesla,
                noise_snr_db=noise_snr_db,
                mask_type=mask_type,
                acceleration=acceleration,
                acs_lines=acs_lines,
                device=device,
            )
            save_dataset_to_h5(synth, h5_path)
        dataset = load_h5_kspace_slice(
            h5_path=h5_path,
            mask_type=mask_type,
            acceleration=acceleration,
            acs_lines=acs_lines,
            fov_x_m=fov_x_m,
            fov_y_m=fov_y_m,
            b0_tesla=b0_tesla,
            device=device,
        )

    trainer = MRIPINNTrainer(dataset=dataset, config=cfg, device=device)
    history, final_maps = trainer.train(epochs=epochs, verbose=True)

    zf_mag = torch.abs(dataset.zero_filled)
    zf_psnr = compute_psnr(zf_mag, dataset.m_mag_gt)
    zf_ssim = compute_ssim(zf_mag, dataset.m_mag_gt)
    final_m = history[-1]

    # Estadistica en el tumor hiper-conductivo (sigma_gt > 1.0 S/m)
    tumor_roi = (dataset.sigma_gt > 1.0) & (dataset.sigma_gt < 1.45)
    if int(tumor_roi.sum().item()) > 0:
        sigma_tumor_pred = float(final_maps["sigma"][tumor_roi].mean().item())
        sigma_tumor_gt = float(dataset.sigma_gt[tumor_roi].mean().item())
    else:
        sigma_tumor_pred = float(final_maps["sigma"].max().item())
        sigma_tumor_gt = 1.20

    print("-" * 88)
    print(f" RESULTADOS FINALES ({epochs} EPOCAS):")
    print(f"   * Zero-Filled Baseline  -> PSNR: {zf_psnr:.2f} dB | SSIM: {zf_ssim:.4f}")
    print(
        f"   * PINN-SENSE Reconst.   -> PSNR: {final_m.psnr_db:.2f} dB (+{final_m.psnr_db - zf_psnr:.2f} dB) | "
        f"SSIM: {final_m.ssim:.4f}"
    )
    print(
        f"   * Conductividad EPT     -> NRMSE: {final_m.nrmse_sigma:.4f} | "
        f"Tumor ROI sigma_pred: {sigma_tumor_pred:.3f} S/m (GT: {sigma_tumor_gt:.3f} S/m)"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
