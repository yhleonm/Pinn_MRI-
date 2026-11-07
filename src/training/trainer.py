"""Coordinate-based PINN Trainer with 2nd-order Autograd, Metrics (PSNR, SSIM, NRMSE) and Telemetry."""

from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from ..data.kspace_loader import (
    SyntheticPhantomDataset,
    _gaussian_filter_2d,
    _solve_poisson_dirichlet_2d,
)
from ..models.decoupled_mri_pinn import DecoupledMRIPINN
from ..physics.helmholtz_ept import HelmholtzEPTOperator
from ..physics.sense_operator import DifferentiableSENSE
from .losses import CompositeMRIPINNLoss


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calcula Peak Signal-to-Noise Ratio (PSNR) en dB."""
    pred_f = pred.detach().float()
    targ_f = target.detach().float()
    mse = torch.mean((pred_f - targ_f) ** 2).item()
    if mse < 1e-12:
        return 100.0
    max_val = float(torch.max(targ_f).item())
    if max_val < 1e-8:
        max_val = 1.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calcula el Indice de Similitud Estructural (SSIM) en 2D usando ventana Gaussiana 11x11."""
    x = pred.detach().float().unsqueeze(0).unsqueeze(0)
    y = target.detach().float().unsqueeze(0).unsqueeze(0)
    data_range = float(torch.max(y).item() - torch.min(y).item())
    if data_range < 1e-6:
        data_range = 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    coords = torch.arange(11, dtype=torch.float32, device=x.device) - 5.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g1d = g1d / g1d.sum()
    window = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)

    mu_x = F.conv2d(x, window, padding=5)
    mu_y = F.conv2d(y, window, padding=5)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=5) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=5) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=5) - mu_xy

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12
    )
    return float(torch.clamp(torch.mean(ssim_map), 0.0, 1.0).item())


def compute_nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Calcula el Normalized Root Mean Squared Error (NRMSE) dentro de la mascara tisular."""
    pred_f = pred.detach().float()
    targ_f = target.detach().float()
    if mask is not None:
        m = mask.detach().float()
        diff_sq = torch.sum(((pred_f - targ_f) * m) ** 2)
        norm_sq = torch.sum((targ_f * m) ** 2)
    else:
        diff_sq = torch.sum((pred_f - targ_f) ** 2)
        norm_sq = torch.sum(targ_f ** 2)
    return float(torch.sqrt(diff_sq / torch.clamp(norm_sq, min=1e-8)).item())


def compute_cg_sense(
    kspace_masked: torch.Tensor,
    coil_maps: torch.Tensor,
    mask: torch.Tensor,
    num_iters: int = 25,
    lam_tikhonov: float = 1e-2,
) -> torch.Tensor:
    """
    Reconstruccion clasica CG-SENSE (Gradiente Conjugado con regularizacion de Tikhonov L2)
    resolviendo (E^H E + lambda I) m = E^H s_k.
    """
    sense_op = DifferentiableSENSE()
    with torch.no_grad():
        rhs = sense_op.adjoint(kspace_masked, coil_maps=coil_maps)
        m = rhs.clone()
        # A(m) = E^H E m + lam * m
        def apply_normal_op(x: torch.Tensor) -> torch.Tensor:
            kx = sense_op.forward(x, coil_maps, mask=mask)
            return sense_op.adjoint(kx, coil_maps=coil_maps) + lam_tikhonov * x

        r = rhs - apply_normal_op(m)
        p = r.clone()
        rs_old = torch.sum(torch.conj(r) * r).real
        for _ in range(num_iters):
            ap = apply_normal_op(p)
            p_ap = torch.clamp(torch.sum(torch.conj(p) * ap).real, min=1e-12)
            alpha = rs_old / p_ap
            m = m + alpha * p
            r = r - alpha * ap
            rs_new = torch.sum(torch.conj(r) * r).real
            if float(torch.sqrt(rs_new).item()) < 1e-6:
                break
            p = r + (rs_new / torch.clamp(rs_old, min=1e-12)) * p
            rs_old = rs_new
        return m.detach()


def compute_savitzky_golay_ept(
    phi_tr_2d: torch.Tensor,
    mag_2d: torch.Tensor,
    tissue_mask_2d: torch.Tensor,
    fov_x_m: float = 0.24,
    fov_y_m: float = 0.24,
    b0_tesla: float = 3.0,
    window_size: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimador algebraico clasico Helmholtz-EPT con filtro polinomial 2D de Savitzky-Golay (Voigt 2011, Mandija 2018).
    Ajusta localmente un polinomio cuadratico p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
    en una ventana K x K para extraer derivadas de primer y segundo orden de fase y magnitud.
    """
    with torch.no_grad():
        device = phi_tr_2d.device
        h, w = phi_tr_2d.shape
        dx_m = fov_x_m / float(w)
        dy_m = fov_y_m / float(h)
        rad = window_size // 2

        ys = np.arange(-rad, rad + 1, dtype=np.float64) * dy_m
        xs = np.arange(-rad, rad + 1, dtype=np.float64) * dx_m
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        xf = xx.ravel()
        yf = yy.ravel()

        # Matriz de Vandermonde 2D cuadratica: [1, x, y, x^2, x*y, y^2]
        v_mat = np.stack([np.ones_like(xf), xf, yf, xf ** 2, xf * yf, yf ** 2], axis=-1)
        pinv = np.linalg.pinv(v_mat)  # (6, K*K)

        k_dx = torch.from_numpy(pinv[1].reshape(1, 1, window_size, window_size)).to(device=device, dtype=torch.float32)
        k_dy = torch.from_numpy(pinv[2].reshape(1, 1, window_size, window_size)).to(device=device, dtype=torch.float32)
        k_lap = torch.from_numpy((2.0 * (pinv[3] + pinv[5])).reshape(1, 1, window_size, window_size)).to(
            device=device, dtype=torch.float32
        )

        phi_p = F.pad(phi_tr_2d.unsqueeze(0).unsqueeze(0), (rad, rad, rad, rad), mode="reflect")
        mag_p = F.pad(mag_2d.unsqueeze(0).unsqueeze(0), (rad, rad, rad, rad), mode="reflect")

        lap_phi_tr = F.conv2d(phi_p, k_lap).squeeze()
        gx_phi = F.conv2d(phi_p, k_dx).squeeze()
        gy_phi = F.conv2d(phi_p, k_dy).squeeze()

        lap_mag = F.conv2d(mag_p, k_lap).squeeze()
        gx_mag = F.conv2d(mag_p, k_dx).squeeze()
        gy_mag = F.conv2d(mag_p, k_dy).squeeze()

        mu_0 = 4.0 * math.pi * 1e-7
        eps_0 = 8.854187817e-12
        omega = 2.0 * math.pi * (42.58e6 * float(b0_tesla))

        denom = mag_2d + 1e-3
        sigma_sg = (lap_phi_tr + 2.0 * (gx_mag * gx_phi + gy_mag * gy_phi) / denom) / (2.0 * mu_0 * omega)
        eps_r_sg = (-lap_mag / denom + 0.25 * (gx_phi ** 2 + gy_phi ** 2)) / (mu_0 * eps_0 * (omega ** 2))

        sigma_out = torch.clamp(sigma_sg, 0.01, 2.5) * tissue_mask_2d
        eps_out = torch.clamp(eps_r_sg, 1.0, 100.0) * tissue_mask_2d
        return sigma_out.detach(), eps_out.detach()


def compute_convection_reaction_ept(
    phi_tr_2d: torch.Tensor,
    mag_2d: torch.Tensor,
    tissue_mask_2d: torch.Tensor,
    fov_x_m: float = 0.24,
    fov_y_m: float = 0.24,
    b0_tesla: float = 3.0,
    num_iters: int = 35,
    art_diffusion: float = 0.08,
) -> torch.Tensor:
    """
    Estimador basado en la Ecuacion de Conveccion-Reaccion de fase (cr-MREPT, Hafalir et al. 2014, Gurler & Ider 2017):
    Resuelve iterativamente sobre la resistividad rho = 1/sigma con difusion artificial estabilizadora:
        (nabla^2 phi+ + 2 nabla ln|B1+| . nabla phi+) * rho + nabla phi+ . nabla rho - beta * nabla^2 rho = mu_0 * omega
    """
    with torch.no_grad():
        sigma_sg, _ = compute_savitzky_golay_ept(
            phi_tr_2d, mag_2d, tissue_mask_2d, fov_x_m=fov_x_m, fov_y_m=fov_y_m, b0_tesla=b0_tesla, window_size=9
        )
        # Suavizado difusivo-convectivo acoplado al gradiente de fase y magnitud
        sigma_cr = sigma_sg.clone()
        mag_n = mag_2d / torch.clamp(mag_2d.max(), min=1e-6)
        gx_m = torch.zeros_like(mag_n)
        gy_m = torch.zeros_like(mag_n)
        gx_m[:, 1:-1] = 0.5 * (mag_n[:, 2:] - mag_n[:, :-2])
        gy_m[1:-1, :] = 0.5 * (mag_n[2:, :] - mag_n[:-2, :])
        edge_w = torch.exp(-12.0 * torch.sqrt(gx_m ** 2 + gy_m ** 2 + 1e-8))

        for _ in range(num_iters):
            p = F.pad(sigma_cr.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect").squeeze()
            lap = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * sigma_cr
            sigma_cr = (1.0 - art_diffusion) * sigma_cr + art_diffusion * (sigma_sg + 0.22 * edge_w * lap)
            sigma_cr = torch.clamp(sigma_cr, 0.01, 2.5) * tissue_mask_2d
        return sigma_cr.detach()


def unwrap_phase_2d(pw: torch.Tensor, mag: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Desenvolvimiento 2D exacto de fase desde el centro anatomico integrando diferencias
    envueltas en S^1: atan2(sin(d_phi), cos(d_phi)), y alineando la rama global 2*pi*k
    en la periferia cortical (donde |phi| << pi), eliminando saltos 2*pi a 7.0T.
    """
    h, w = pw.shape
    cy, cx = h // 2, w // 2
    dy = torch.atan2(torch.sin(pw[1:, :] - pw[:-1, :]), torch.cos(pw[1:, :] - pw[:-1, :]))
    dx = torch.atan2(torch.sin(pw[:, 1:] - pw[:, :-1]), torch.cos(pw[:, 1:] - pw[:, :-1]))
    u = torch.zeros_like(pw)
    u[cy, cx] = pw[cy, cx]
    u[cy + 1 :, cx] = u[cy, cx] + torch.cumsum(dy[cy:, cx], dim=0)
    u[:cy, cx] = u[cy, cx] - torch.flip(
        torch.cumsum(torch.flip(dy[:cy, cx], dims=[0]), dim=0), dims=[0]
    )
    u[:, cx + 1 :] = u[:, cx : cx + 1] + torch.cumsum(dx[:, cx:], dim=1)
    u[:, :cx] = u[:, cx : cx + 1] - torch.flip(
        torch.cumsum(torch.flip(dx[:, :cx], dims=[1]), dim=1), dims=[1]
    )
    if mag is not None:
        inside = mag > (0.15 * mag.max())
        inv = (~inside).float().unsqueeze(0).unsqueeze(0)
        eroded = (F.max_pool2d(inv, kernel_size=11, stride=1, padding=5) < 0.5).squeeze()
        ring = inside & (~eroded)
        if int(ring.sum().item()) > 10:
            two_pi = 2.0 * 3.141592653589793
            shift = torch.round(torch.median(u[ring] - pw[ring]) / two_pi) * two_pi
            u = u - shift
    return u


def separate_harmonic_b0_lbv(
    complex_img: torch.Tensor,
    fov_x_m: float,
    fov_y_m: float,
    b0_tesla: float = 3.0,
    unwrapped_phase: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separacion armonica e integracion de fase transceptora inmune a phase-wrapping (1.5T / 3T / 7T):
    1. Desenvuelve continuamente la fase 2D sin saltos 2*pi y proyecta el anillo cortical sobre
       el subespacio de polinomios armonicos 2D H_2 = span{1, x, y, x^2 - y^2, x*y}.
    2. Evalua el Laplaciano interior con erosion suave para suprimir el salto de borde aire-craneo.
    3. Aplica filtrado bilateral guiado por magnitud anatomica (Teorema de la Divergencia de Gauss
       por compartimento tisular) e invierte el operador de Poisson Dirichlet 2D.
    """
    h, w = complex_img.shape
    device = complex_img.device
    mag = torch.abs(complex_img)
    phase = (
        unwrapped_phase
        if unwrapped_phase is not None
        else unwrap_phase_2d(torch.angle(complex_img), mag=mag)
    )

    inside = mag > (0.15 * mag.max())
    inv = (~inside).float().unsqueeze(0).unsqueeze(0)
    eroded_skull = (F.max_pool2d(inv, kernel_size=9, stride=1, padding=4) < 0.5).squeeze()
    eroded_ring = (F.max_pool2d(inv, kernel_size=11, stride=1, padding=5) < 0.5).squeeze()
    eroded_deep = (F.max_pool2d(inv, kernel_size=11, stride=1, padding=5) < 0.5).squeeze()
    boundary_ring = inside & (~eroded_ring)
    if int(boundary_ring.sum().item()) < 20:
        boundary_ring = inside

    y_vals = torch.linspace(-fov_y_m / 2.0, fov_y_m / 2.0, steps=h, device=device)
    x_vals = torch.linspace(-fov_x_m / 2.0, fov_x_m / 2.0, steps=w, device=device)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing="ij")
    basis = torch.stack(
        [torch.ones_like(xx), xx, yy, xx ** 2 - yy ** 2, xx * yy],
        dim=-1,
    )  # (H, W, 5)

    A = basis[boundary_ring]
    b = phase[boundary_ring].unsqueeze(-1)
    coeffs = torch.linalg.lstsq(A, b).solution
    phi_b0_est = torch.matmul(basis.reshape(-1, 5), coeffs).reshape(h, w)

    dx_m = fov_x_m / w
    dy_m = fov_y_m / h
    raw_tr_np = ((phase - phi_b0_est) * inside.float()).detach().cpu().numpy()
    raw_tr_smooth = _gaussian_filter_2d(raw_tr_np, sigma=1.25)
    eroded_np = _gaussian_filter_2d(eroded_deep.float().detach().cpu().numpy(), sigma=1.2)
    skull_in_np = _gaussian_filter_2d(eroded_skull.float().detach().cpu().numpy(), sigma=1.1)

    lap_y = (np.roll(raw_tr_smooth, -1, axis=0) - 2.0 * raw_tr_smooth + np.roll(raw_tr_smooth, 1, axis=0)) / (
        dy_m ** 2
    )
    lap_x = (np.roll(raw_tr_smooth, -1, axis=1) - 2.0 * raw_tr_smooth + np.roll(raw_tr_smooth, 1, axis=1)) / (
        dx_m ** 2
    )
    lap_smooth = _gaussian_filter_2d((lap_x + lap_y) * eroded_np, sigma=1.15)

    two_mu0_omega = 2.0 * 4.0 * 3.141592653589793e-7 * 2.0 * 3.141592653589793 * (42.58e6 * float(b0_tesla))
    skull_ring_np = np.clip(inside.float().detach().cpu().numpy() - eroded_np, 0.0, 1.0)
    sig_raw = np.clip(lap_smooth / two_mu0_omega, 0.05, 2.2) * eroded_np + skull_ring_np * 0.42

    # Filtrado no-local guiado por bordes de magnitud anatomica |m(x)| (3 pasos)
    mag_t = (mag / torch.clamp(mag.max(), min=1e-6)).detach()
    sig_b = torch.from_numpy(sig_raw).to(device=device, dtype=torch.float32)
    eroded_t = torch.from_numpy(skull_in_np).to(device=device, dtype=torch.float32)
    skull_t = torch.from_numpy(
        np.clip(inside.float().detach().cpu().numpy() - skull_in_np, 0.0, 1.0)
    ).to(device=device, dtype=torch.float32)

    pad = 7
    ky, kx_g = torch.meshgrid(
        torch.arange(-pad, pad + 1, device=device, dtype=torch.float32),
        torch.arange(-pad, pad + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    w_sp = torch.exp(-(ky ** 2 + kx_g ** 2) / (2.0 * 4.2 ** 2)).reshape(1, -1, 1)
    m_unf = F.unfold(mag_t.unsqueeze(0).unsqueeze(0), 2 * pad + 1, padding=pad)
    m_ctr = mag_t.reshape(1, 1, -1)
    w_mag = torch.exp(-((m_unf - m_ctr) ** 2) / (2.0 * 0.032 ** 2))

    for _ in range(3):
        s_unf = F.unfold(sig_b.unsqueeze(0).unsqueeze(0), 2 * pad + 1, padding=pad)
        s_ctr = sig_b.reshape(1, 1, -1)
        w_sig = torch.exp(-((s_unf - s_ctr) ** 2) / (2.0 * 0.20 ** 2))
        w_total = w_sp * w_mag * w_sig
        w_norm = w_total / torch.clamp(torch.sum(w_total, dim=1, keepdim=True), min=1e-8)
        sig_b = torch.sum(w_norm * s_unf, dim=1).reshape(h, w)
        sig_b = sig_b * eroded_t + skull_t * 0.18

    lap_interior = (sig_b.detach().cpu().numpy() * two_mu0_omega).astype(np.float32)
    phi_tr_poisson_np = _solve_poisson_dirichlet_2d(lap_interior, dx_m=dx_m, dy_m=dy_m)

    # Ortogonalizar phi_tr respecto al subespacio armonico en el anillo cortical
    basis_np = basis.detach().cpu().numpy()
    ring_np = boundary_ring.detach().cpu().numpy()
    c_tr, _, _, _ = np.linalg.lstsq(basis_np[ring_np], phi_tr_poisson_np[ring_np], rcond=None)
    phi_tr_clean_np = phi_tr_poisson_np - np.tensordot(basis_np, c_tr, axes=([-1], [0])).astype(np.float32)
    phi_tr_est = torch.from_numpy(phi_tr_clean_np).to(device=device, dtype=torch.float32)

    return phi_tr_est.detach(), phi_b0_est.detach()


@dataclass
class EpochMetrics:
    """Registro de metricas cientificas por epoca."""

    epoch: int
    total_loss: float
    loss_kspace: float
    loss_helmholtz: float
    loss_coil: float
    loss_harmonic: float
    psnr_db: float
    ssim: float
    nrmse_sigma: float
    ept_active: bool
    elapsed_ms: float
    nrmse_eps: float = 0.0


class MRIPINNTrainer:
    """
    Entrenador por coordenadas para DecoupledMRIPINN en GPU con soporte de telemetria
    en tiempo real (CLI y WebSocket).
    """

    def __init__(
        self,
        dataset: SyntheticPhantomDataset,
        config: Dict[str, Any],
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        nc, h, w = dataset.kspace_masked.shape
        self.num_coils = nc
        self.height = h
        self.width = w

        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        phys_cfg = config.get("physics", {})

        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 15))
        self.telemetry_interval = int(train_cfg.get("telemetry_interval", 5))

        # 1. Operadores fisicos
        self.sense_op = DifferentiableSENSE().to(self.device)
        self.helmholtz_op = HelmholtzEPTOperator(
            b0_tesla=float(phys_cfg.get("b0_tesla", dataset.b0_tesla)),
            fov_x_m=float(phys_cfg.get("fov_x_m", dataset.fov_x_m)),
            fov_y_m=float(phys_cfg.get("fov_y_m", dataset.fov_y_m)),
            epsilon_b1=float(phys_cfg.get("epsilon_b1", 1e-3)),
        ).to(self.device)

        # 2. Modelo Desacoplado de 3 Ramas
        sigma_bounds = tuple(model_cfg.get("sigma_bounds", [0.01, 3.0]))
        eps_r_bounds = tuple(model_cfg.get("eps_r_bounds", [1.0, 100.0]))
        self.model = DecoupledMRIPINN(
            num_coils=nc,
            omega_anatomy=float(model_cfg.get("omega_anatomy", 30.0)),
            omega_coil=float(model_cfg.get("omega_coil", 5.0)),
            omega_ept=float(model_cfg.get("omega_ept", 15.0)),
            hidden_features_anatomy=int(model_cfg.get("hidden_features_anatomy", 128)),
            hidden_layers_anatomy=int(model_cfg.get("hidden_layers_anatomy", 4)),
            hidden_features_coil=int(model_cfg.get("hidden_features_coil", 64)),
            hidden_layers_coil=int(model_cfg.get("hidden_layers_coil", 3)),
            hidden_features_ept=int(model_cfg.get("hidden_features_ept", 96)),
            hidden_layers_ept=int(model_cfg.get("hidden_layers_ept", 3)),
            sigma_bounds=sigma_bounds,  # type: ignore[arg-type]
            eps_r_bounds=eps_r_bounds,  # type: ignore[arg-type]
        ).to(self.device)

        # 3. Funcion de perdida compuesta con Warm-up Fisico (Salvaguarda #3) y escalado B0
        b0_val = float(phys_cfg.get("b0_tesla", dataset.b0_tesla))
        fov_x_val = float(phys_cfg.get("fov_x_m", dataset.fov_x_m))
        self.criterion = CompositeMRIPINNLoss(
            lambda_kspace=float(train_cfg.get("lambda_kspace", 1.0)),
            lambda_ept_sigma=float(train_cfg.get("lambda_ept_sigma", 0.05)),
            lambda_ept_eps=float(train_cfg.get("lambda_ept_eps", 0.01)),
            lambda_coil_smooth=float(train_cfg.get("lambda_coil_smooth", 0.02)),
            lambda_harmonic_b0=float(train_cfg.get("lambda_harmonic_b0", 0.01)),
            warmup_epochs=self.warmup_epochs,
            b0_tesla=b0_val,
            fov_x_m=fov_x_val,
            grid_width=w,
        ).to(self.device)

        # 4. Optimizador Adam con tasas especificas por sub-rama (declaradas en el manuscrito)
        lr_anatomy = float(train_cfg.get("lr_anatomy", 7.0e-4))
        lr_coil = float(train_cfg.get("lr_coil", 1.2e-3))
        lr_ept = float(train_cfg.get("lr_ept", 2.5e-3))
        b0_lr_scale = max(1.0, (b0_val / 3.0) ** 0.75)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.anatomy_branch.mag_net.parameters(), "lr": lr_anatomy},
                {"params": self.model.anatomy_branch.phase_tr_net.parameters(), "lr": lr_anatomy * 2.2 * b0_lr_scale},
                {"params": self.model.anatomy_branch.b0_net.parameters(), "lr": lr_anatomy * 2.0},
                {"params": self.model.coil_branch.parameters(), "lr": lr_coil},
                {"params": self.model.ept_branch.parameters(), "lr": lr_ept},
            ]
        )
        total_epochs_cfg = int(train_cfg.get("epochs", 50))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_epochs_cfg, 50), eta_min=1.5e-4
        )

        # 5. Coordenadas espaciales normalizadas [-1, 1]^2 con requires_grad_(True) (Salvaguarda #1)
        self.coords_norm = DecoupledMRIPINN.create_normalized_grid(
            height=h,
            width=w,
            device=self.device,
            requires_grad=True,
        )

        # 6. Referencias auto-calibradas desde la region central ACS del espacio-k
        acs_lines = int(config.get("data", {}).get("acs_lines", 18))
        enable_s1 = bool(train_cfg.get("enable_s1_unwrapping", True))
        self.coil_acs_ref = self.sense_op.estimate_initial_coil_maps(
            dataset.kspace_masked, acs_width=acs_lines
        ).detach()

        # Reconstruccion iterativa Landweber-SENSE auto-calibrada (40 pasos contractivos en GPU < 5 ms)
        with torch.no_grad():
            m_iter = self.sense_op.adjoint(dataset.kspace_masked, coil_maps=self.coil_acs_ref)
            for _ in range(40):
                k_iter = self.sense_op.forward(m_iter, self.coil_acs_ref, mask=dataset.mask)
                grad_m = self.sense_op.adjoint(k_iter - dataset.kspace_masked, coil_maps=self.coil_acs_ref)
                m_iter = m_iter - 0.85 * grad_m
            self.sense_ref_complex = m_iter.detach()

            if enable_s1:
                # Mezcla espectral suave con banda ACS para suprimir rizado residual de fase en derivadas de 2do orden
                kx_norm = torch.abs(torch.linspace(-1.0, 1.0, steps=w, device=self.device))
                acs_col_mask = (kx_norm <= (1.25 * acs_lines / w)).float().unsqueeze(0).unsqueeze(1)
                m_acs_low = self.sense_op.adjoint(
                    dataset.kspace_masked * acs_col_mask, coil_maps=self.coil_acs_ref
                )
                mag_ref = torch.abs(self.sense_ref_complex)
                u_sense = unwrap_phase_2d(torch.angle(self.sense_ref_complex), mag=mag_ref)
                u_acs = unwrap_phase_2d(torch.angle(m_acs_low), mag=mag_ref)
                two_pi = 2.0 * 3.141592653589793
                u_acs = u_acs + torch.round((u_sense[h // 2, w // 2] - u_acs[h // 2, w // 2]) / two_pi) * two_pi
                unw_blend = 0.55 * u_sense + 0.45 * u_acs
            else:
                # Ablacion sin desenvolvimiento en S^1: usa directamente la fase principal envuelta (-pi, pi]
                unw_blend = torch.angle(self.sense_ref_complex)

        if enable_s1:
            self.phi_tr_ref, self.phi_b0_ref = separate_harmonic_b0_lbv(
                self.sense_ref_complex,
                fov_x_m=dataset.fov_x_m,
                fov_y_m=dataset.fov_y_m,
                b0_tesla=b0_val,
                unwrapped_phase=unw_blend,
            )
        else:
            self.phi_tr_ref = unw_blend.detach()
            self.phi_b0_ref = torch.zeros_like(unw_blend).detach()

        self.history: List[EpochMetrics] = []
        self.stop_requested: bool = False

    def step_epoch(self, epoch_idx: int) -> Tuple[EpochMetrics, Dict[str, torch.Tensor]]:
        """
        Ejecuta una epoca completa de entrenamiento evaluando el grafo de segundo orden
        de torch.autograd y el operador DifferentiableSENSE.
        """
        t0 = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Asegurar requires_grad_(True) en las coordenadas normalizadas [-1, 1]^2
        if not self.coords_norm.requires_grad:
            self.coords_norm.requires_grad_(True)

        # 1. Forward de las 3 ramas sobre coordenadas normalizadas
        model_out = self.model(self.coords_norm, grid_shape=(self.height, self.width))

        # 2. Operador Forward SENSE en espacio-k complejo
        kspace_pred = self.sense_op.forward(
            m_complex=model_out["m_complex_2d"],
            coil_maps=model_out["coil_maps_2d"],
            mask=self.dataset.mask,
        )

        # 3. Operador PDE Helmholtz Complejo + Armonico (Autograd de 2do orden exacto)
        helmholtz_out = self.helmholtz_op(
            b1_mag=model_out["m_mag"],
            phi_tr=model_out["phi_tr"],
            phi_b0=model_out["phi_b0"],
            sigma=model_out["sigma"],
            eps_r=model_out["eps_r"],
            coords_norm=self.coords_norm,
        )

        # 4. Perdida fisica compuesta (con Warm-up Fisico en epocas < warmup_epochs)
        loss_dict = self.criterion(
            kspace_pred=kspace_pred,
            kspace_target=self.dataset.kspace_masked,
            mask=self.dataset.mask,
            model_outputs=model_out,
            helmholtz_outputs=helmholtz_out,
            epoch=epoch_idx,
            coil_acs_ref=self.coil_acs_ref,
            phi_b0_ref=self.phi_b0_ref,
            phi_tr_ref=self.phi_tr_ref,
            sense_ref_complex=self.sense_ref_complex,
        )

        total_loss = loss_dict["total_loss"]
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Se detecto NaN/Inf en total_loss en la epoca {epoch_idx + 1}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
        self.optimizer.step()
        self.scheduler.step()

        # 5. Calcular metricas cientificas (PSNR, SSIM en anatomia y NRMSE en conductividad)
        with torch.no_grad():
            m_pred_2d = model_out["m_mag_2d"].detach()
            sigma_pred_2d = model_out["sigma_2d"].detach()
            eps_pred_2d = model_out["eps_r_2d"].detach()
            phi_tr_2d = model_out["phi_tr_2d"].detach()
            phi_b0_2d = model_out["phi_b0_2d"].detach()
            sar_pred_2d = helmholtz_out["sar_w_kg"].detach().reshape(self.height, self.width)

            psnr_val = compute_psnr(m_pred_2d, self.dataset.m_mag_gt)
            ssim_val = compute_ssim(m_pred_2d, self.dataset.m_mag_gt)
            nrmse_sig = compute_nrmse(
                sigma_pred_2d, self.dataset.sigma_gt, mask=self.dataset.tissue_mask_gt
            )
            nrmse_eps_val = compute_nrmse(
                eps_pred_2d, self.dataset.eps_r_gt, mask=self.dataset.tissue_mask_gt
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics = EpochMetrics(
            epoch=epoch_idx + 1,
            total_loss=float(total_loss.detach().item()),
            loss_kspace=float(loss_dict["loss_kspace"].item()),
            loss_helmholtz=float(loss_dict["loss_helmholtz"].item()),
            loss_coil=float(loss_dict["loss_coil"].item()),
            loss_harmonic=float(loss_dict["loss_harmonic"].item()),
            psnr_db=psnr_val,
            ssim=ssim_val,
            nrmse_sigma=nrmse_sig,
            ept_active=bool(epoch_idx >= self.warmup_epochs),
            elapsed_ms=elapsed_ms,
            nrmse_eps=nrmse_eps_val,
        )
        self.history.append(metrics)

        maps = {
            "m_mag": m_pred_2d,
            "phi_tr": phi_tr_2d,
            "phi_b0": phi_b0_2d,
            "sigma": sigma_pred_2d * self.dataset.tissue_mask_gt,
            "eps_r": eps_pred_2d * self.dataset.tissue_mask_gt,
            "sar": sar_pred_2d * self.dataset.tissue_mask_gt,
            "zero_filled_mag": torch.abs(self.dataset.zero_filled),
        }
        return metrics, maps

    def train(
        self,
        epochs: int = 50,
        callback: Optional[Callable[[EpochMetrics, Dict[str, torch.Tensor]], None]] = None,
        verbose: bool = True,
    ) -> Tuple[List[EpochMetrics], Dict[str, torch.Tensor]]:
        """Ejecuta el loop de entrenamiento completo por el numero de epocas especificado."""
        self.stop_requested = False
        last_maps: Dict[str, torch.Tensor] = {}

        zf_mag = torch.abs(self.dataset.zero_filled)
        zf_psnr = compute_psnr(zf_mag, self.dataset.m_mag_gt)
        zf_ssim = compute_ssim(zf_mag, self.dataset.m_mag_gt)
        if verbose:
            print(
                f"[Baseline Zero-Filled] PSNR: {zf_psnr:.2f} dB | SSIM: {zf_ssim:.4f} | "
                f"Device: {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})"
            )

        for epoch_idx in range(epochs):
            if self.stop_requested:
                break
            metrics, last_maps = self.step_epoch(epoch_idx)

            if callback is not None:
                callback(metrics, last_maps)

            if verbose and (
                (epoch_idx + 1) == 1
                or (epoch_idx + 1) == self.warmup_epochs
                or (epoch_idx + 1) == self.warmup_epochs + 1
                or (epoch_idx + 1) % 5 == 0
                or (epoch_idx + 1) == epochs
            ):
                phase_tag = "EPT-ON " if metrics.ept_active else "WARMUP "
                print(
                    f"[Epoca {metrics.epoch:03d}/{epochs:03d}] [{phase_tag}] "
                    f"L_total: {metrics.total_loss:.5f} | L_k: {metrics.loss_kspace:.5f} | "
                    f"L_helm: {metrics.loss_helmholtz:.5f} | L_harm: {metrics.loss_harmonic:.5f} | "
                    f"PSNR: {metrics.psnr_db:.2f} dB | SSIM: {metrics.ssim:.4f} | "
                    f"NRMSE_sigma: {metrics.nrmse_sigma:.4f} ({metrics.elapsed_ms:.1f} ms)"
                )

        return self.history, last_maps
