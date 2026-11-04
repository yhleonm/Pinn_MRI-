"""2D Sub-sampling Masks (Cartesian, Radial, Poisson-Disc) with Autocalibration (ACS)."""

from __future__ import annotations
import math
import numpy as np
import torch


def _cartesian_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    acs_lines: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mascara cartesiana 1D de densidad variable a lo largo de codificacion de fase (columnas)."""
    if rng is None:
        rng = np.random.default_rng(42)

    mask = np.zeros((height, width), dtype=np.float32)
    center = width // 2
    half_acs = max(2, acs_lines // 2)
    acs_start = max(0, center - half_acs)
    acs_end = min(width, center + half_acs)
    mask[:, acs_start:acs_end] = 1.0

    target_lines = max(acs_end - acs_start, int(round(width / max(1, acceleration))))
    remaining_lines = max(0, target_lines - (acs_end - acs_start))

    if remaining_lines > 0:
        # Muestreo estratificado de densidad variable (evita huecos mayores al factor R de SENSE)
        left_cols = np.arange(1, acs_start)
        right_cols = np.arange(acs_end, width - 1)
        n_left = remaining_lines // 2
        n_right = remaining_lines - n_left

        if n_left > 0 and len(left_cols) > 0:
            # Densidad mayor cerca de ACS (potencia 0.75 hacia el centro)
            u_l = np.linspace(0.0, 1.0, n_left) ** 0.85
            idx_l = np.unique(
                (len(left_cols) - 1 - np.round(u_l * (len(left_cols) - 1)).astype(int)).clip(
                    0, len(left_cols) - 1
                )
            )
            mask[:, left_cols[idx_l]] = 1.0

        if n_right > 0 and len(right_cols) > 0:
            u_r = np.linspace(0.0, 1.0, n_right) ** 0.85
            idx_r = np.unique(
                np.round(u_r * (len(right_cols) - 1)).astype(int).clip(0, len(right_cols) - 1)
            )
            mask[:, right_cols[idx_r]] = 1.0

    return mask


def _radial_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    acs_lines: int = 18,
) -> np.ndarray:
    """Mascara radial de angulo aureo (Golden-Angle) con disco central ACS."""
    mask = np.zeros((height, width), dtype=np.float32)
    cy, cx = height / 2.0, width / 2.0

    # Numero de rayos proporcional a aceleracion objetivo
    num_spokes = max(12, int(round((height * math.pi / 2.0) / max(1, acceleration))))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    max_r = math.hypot(cy, cx)
    num_samples = int(max_r * 2.5)
    radii = np.linspace(-max_r, max_r, num_samples)

    for s in range(num_spokes):
        theta = s * golden_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        xs = np.round(cx + radii * cos_t).astype(int)
        ys = np.round(cy + radii * sin_t).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[valid], xs[valid]] = 1.0

    # Region central de autocalibracion (ACS)
    yy, xx = np.ogrid[:height, :width]
    r_acs = max(4.0, acs_lines / 2.0)
    acs_disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_acs ** 2)
    mask[acs_disc] = 1.0
    return mask


def _poisson_disc_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    acs_lines: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mascara 2D de densidad variable tipo Poisson-Disc con ventana ACS central."""
    if rng is None:
        rng = np.random.default_rng(42)

    mask = np.zeros((height, width), dtype=np.float32)
    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    r_norm = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)

    # Region ACS central cuadrada/eliptica
    r_acs = max(4.0, acs_lines / 2.0)
    acs_region = (np.abs(yy - cy) <= r_acs) & (np.abs(xx - cx) <= r_acs)
    mask[acs_region] = 1.0

    target_total = int(round((height * width) / max(1, acceleration)))
    current_count = int(mask.sum())
    needed = max(0, target_total - current_count)

    if needed > 0:
        weights = np.clip((1.0 - 0.65 * np.clip(r_norm, 0.0, 1.4)) ** 2.2, 1e-4, 1.0)
        weights[acs_region] = 0.0
        flat_w = weights.ravel()
        flat_w /= flat_w.sum()
        chosen_idx = rng.choice(
            height * width,
            size=min(needed, int((~acs_region).sum())),
            replace=False,
            p=flat_w,
        )
        mask.ravel()[chosen_idx] = 1.0

    return mask


def create_sampling_mask(
    height: int,
    width: int,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    seed: int = 42,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Crea la mascara binaria 2D P(k) en el dispositivo indicado.

    Args:
        height: Alto de la matriz (H).
        width: Ancho de la matriz (W).
        mask_type: 'cartesian' | 'radial' | 'poisson'.
        acceleration: Factor de sub-muestreo R (ej. 4 o 8).
        acs_lines: Lineas centrales de autocalibracion.
        seed: Semilla reproducible.
        device: Dispositivo PyTorch ('cuda' o 'cpu').

    Returns:
        Tensor float32 de shape (H, W) con valores en {0.0, 1.0}.
    """
    if acceleration <= 1:
        return torch.ones((height, width), dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    mtype = mask_type.lower().strip()

    if mtype == "cartesian":
        mask_np = _cartesian_mask(height, width, acceleration, acs_lines, rng)
    elif mtype == "radial":
        mask_np = _radial_mask(height, width, acceleration, acs_lines)
    elif mtype in ("poisson", "poisson_disc", "poisson-disc"):
        mask_np = _poisson_disc_mask(height, width, acceleration, acs_lines, rng)
    else:
        raise ValueError(f"Tipo de mascara no soportado: {mask_type}")

    return torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
