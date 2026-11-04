"""Composite Loss Function for k-space SENSE + Complex Helmholtz EPT + Harmonic B0."""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeMRIPINNLoss(nn.Module):
    """
    Funcion de perdida fisica compuesta para DecoupledMRIPINN:
      1. L_kspace: Fidelidad en espacio-k complejo sub-muestreado por bobina.
      2. L_coil: Suavidad electromagnetica de mapas de sensibilidad ||nabla C_j||_2^2 + anclaje ACS.
      3. L_harmonic: Condicion de Laplace sobre la fase parasita de fondo ||nabla^2 phi_B0||_2^2.
      4. L_helmholtz: Residuos PDE de conductividad (R_sigma) y permitividad (R_eps) ponderados
         por tissue_weight y con warm-up progresivo (lambda_ept = 0 durante las primeras 15 epocas).
    """

    def __init__(
        self,
        lambda_kspace: float = 1.0,
        lambda_ept_sigma: float = 0.05,
        lambda_ept_eps: float = 0.01,
        lambda_coil_smooth: float = 0.02,
        lambda_harmonic_b0: float = 0.01,
        warmup_epochs: int = 15,
        b0_tesla: float = 3.0,
        fov_x_m: float = 0.24,
        grid_width: int = 128,
    ) -> None:
        super().__init__()
        self.lambda_kspace = float(lambda_kspace)
        self.lambda_ept_sigma = float(lambda_ept_sigma)
        self.lambda_ept_eps = float(lambda_ept_eps)
        self.lambda_coil_smooth = float(lambda_coil_smooth)
        self.lambda_harmonic_b0 = float(lambda_harmonic_b0)
        self.warmup_epochs = int(warmup_epochs)
        self.b0_tesla = float(b0_tesla)
        dx_m = float(fov_x_m) / float(grid_width)
        mu_0 = 4.0 * 3.141592653589793e-7
        omega = 2.0 * 3.141592653589793 * 42.58e6 * self.b0_tesla
        self.lap_to_sigma_scale = (dx_m ** 2) * 2.0 * mu_0 * omega

    @staticmethod
    def _huber_tv_smoothness(field_2d: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
        """Variacion Total Suave (Huber-TV) para suprimir oscilaciones espurias de frontera."""
        dx = field_2d[..., :, 1:] - field_2d[..., :, :-1]
        dy = field_2d[..., 1:, :] - field_2d[..., :-1, :]
        tv_x = F.smooth_l1_loss(dx, torch.zeros_like(dx), beta=delta, reduction="mean")
        tv_y = F.smooth_l1_loss(dy, torch.zeros_like(dy), beta=delta, reduction="mean")
        return tv_x + tv_y

    @staticmethod
    def _spatial_tv_smoothness(field_2d: torch.Tensor) -> torch.Tensor:
        """Calcula la norma L2 del gradiente espacial en la grilla 2D/3D."""
        dy = field_2d[..., 1:, :] - field_2d[..., :-1, :]
        dx = field_2d[..., :, 1:] - field_2d[..., :, :-1]
        return torch.mean(dx ** 2) + torch.mean(dy ** 2)

    @staticmethod
    def _edge_aware_smoothness(
        prop_2d: torch.Tensor,
        anat_mag_2d: torch.Tensor,
        edge_alpha: float = 15.0,
    ) -> torch.Tensor:
        """Regularizador espacial guiado por bordes anatomicos (sin gradiente hacia anatomia)."""
        anat_det = anat_mag_2d.detach()
        gx_anat = torch.abs(anat_det[:, 1:] - anat_det[:, :-1])
        gy_anat = torch.abs(anat_det[1:, :] - anat_det[:-1, :])
        wx = torch.exp(-edge_alpha * gx_anat)
        wy = torch.exp(-edge_alpha * gy_anat)

        dx_p = (prop_2d[:, 1:] - prop_2d[:, :-1]) ** 2
        dy_p = (prop_2d[1:, :] - prop_2d[:-1, :]) ** 2
        return torch.mean(wx * dx_p) + torch.mean(wy * dy_p)

    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        helmholtz_outputs: Dict[str, torch.Tensor],
        epoch: int,
        coil_acs_ref: Optional[torch.Tensor] = None,
        phi_b0_ref: Optional[torch.Tensor] = None,
        phi_tr_ref: Optional[torch.Tensor] = None,
        sense_ref_complex: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula la perdida total y el desglose de componentes fisicas.
        """
        # 1. Perdida en espacio-k complejo sub-muestreado: || P(k)*F{C_j * m} - s_j(k) ||_2^2
        diff_k = (kspace_pred - kspace_target) * mask.unsqueeze(0)
        active_elements = torch.clamp(torch.sum(mask) * kspace_pred.shape[0], min=1.0)
        loss_kspace = torch.sum(torch.abs(diff_k) ** 2) / active_elements

        # Consistencia espacial auxiliar con el operador Adjoint/CG-SENSE auto-calibrado
        if sense_ref_complex is not None:
            m_mag_2d = model_outputs["m_mag_2d"]
            phi_tr_2d = model_outputs["phi_tr_2d"]
            phi_b0_2d = model_outputs["phi_b0_2d"]
            ref_mag = torch.abs(sense_ref_complex)
            fg_weight = (ref_mag > 0.08 * ref_mag.max()).float()
            bg_weight = 1.0 - fg_weight
            loss_spatial_mag = torch.mean((m_mag_2d - ref_mag) ** 2) + 0.5 * torch.mean(
                bg_weight * (m_mag_2d ** 2)
            )
            if phi_tr_ref is not None:
                # Anclaje de fase continua en espacio complejo exp(i*(phi_tr + phi_B0)) inmune a saltos 2*pi
                ref_phase = torch.angle(sense_ref_complex)
                phase_total_pred = phi_tr_2d + phi_b0_2d
                loss_complex_phase = torch.sum(
                    (1.0 - torch.cos(phase_total_pred - ref_phase)) * fg_weight
                ) / torch.clamp(fg_weight.sum(), min=1.0)

                # Consistencia con la fase transceptora desenvuelta phi_tr_ref y su Laplaciano autograd
                b0_norm = max(self.b0_tesla / 3.0, 1.0)
                diff_phi = (phi_tr_2d - phi_tr_ref) / b0_norm
                d2x_ref = phi_tr_ref[1:-1, 2:] - 2.0 * phi_tr_ref[1:-1, 1:-1] + phi_tr_ref[1:-1, :-2]
                d2y_ref = phi_tr_ref[2:, 1:-1] - 2.0 * phi_tr_ref[1:-1, 1:-1] + phi_tr_ref[:-2, 1:-1]
                sigma_ref_interior = torch.clamp(
                    (d2x_ref + d2y_ref) / max(self.lap_to_sigma_scale, 1e-6), 0.01, 2.5
                ).detach()

                sigma_aut_2d = helmholtz_outputs["sigma_lap_autograd"].reshape_as(phi_tr_2d)[1:-1, 1:-1]
                loss_sobolev_h2 = torch.mean((sigma_aut_2d - sigma_ref_interior) ** 2)

                # Regularizacion de Tikhonov suave sobre el gradiente espacial de phi_tr cuando B0 >= 7.0T
                if self.b0_tesla >= 7.0:
                    gx_diff = (phi_tr_2d[:, 1:] - phi_tr_2d[:, :-1]) - (
                        phi_tr_ref[:, 1:] - phi_tr_ref[:, :-1]
                    )
                    gy_diff = (phi_tr_2d[1:, :] - phi_tr_2d[:-1, :]) - (
                        phi_tr_ref[1:, :] - phi_tr_ref[:-1, :]
                    )
                    tikhonov_7t = 0.25 * (torch.mean(gx_diff ** 2) + torch.mean(gy_diff ** 2))
                else:
                    tikhonov_7t = 0.0

                loss_spatial_phase = (
                    0.45 * loss_complex_phase
                    + 0.85 * torch.mean(diff_phi ** 2)
                    + 0.65 * loss_sobolev_h2
                    + tikhonov_7t
                )
            else:
                ref_phase = torch.angle(sense_ref_complex)
                loss_spatial_phase = torch.sum(
                    (1.0 - torch.cos((phi_tr_2d + phi_b0_2d) - ref_phase)) * fg_weight
                ) / torch.clamp(fg_weight.sum(), min=1.0)
            loss_kspace = 0.25 * loss_kspace + 0.85 * loss_spatial_mag + 1.50 * loss_spatial_phase

        # 2. Restriccion PDE de suavidad electromagnetica de bobinas: ||nabla C_j(x)||_2^2
        coil_maps_2d = model_outputs["coil_maps_2d"]  # (Nc, H, W)
        loss_coil_grad = self._spatial_tv_smoothness(coil_maps_2d.real) + self._spatial_tv_smoothness(
            coil_maps_2d.imag
        )
        if coil_acs_ref is not None:
            loss_coil_acs = torch.mean(torch.abs(coil_maps_2d - coil_acs_ref) ** 2)
            loss_coil = loss_coil_grad + 0.8 * loss_coil_acs
        else:
            loss_coil = loss_coil_grad

        # 3. Separacion armonica de fase de fondo: ||nabla^2 phi_B0(x)||_2^2
        r_harmonic = helmholtz_outputs["r_harmonic"]
        loss_harmonic = torch.mean(r_harmonic ** 2)
        if phi_b0_ref is not None:
            phi_b0_2d = model_outputs["phi_b0_2d"]
            loss_harmonic = loss_harmonic + 0.5 * torch.mean((phi_b0_2d - phi_b0_ref) ** 2)

        # 4. Residuos PDE de Helmholtz (EPT) con Warm-up Fisico (Salvaguarda #3: lambda_ept = 0 en epocas < 15)
        if epoch < self.warmup_epochs:
            eff_lambda_sigma = 0.0
            eff_lambda_eps = 0.0
        else:
            eff_lambda_sigma = self.lambda_ept_sigma
            eff_lambda_eps = self.lambda_ept_eps

        r_sigma = helmholtz_outputs["r_sigma"]
        r_eps_rel = helmholtz_outputs["r_eps_rel"]
        tissue_weight = helmholtz_outputs["tissue_weight"]
        tissue_count = torch.clamp(torch.sum(tissue_weight), min=1.0)

        # Huber / Smooth L1 sobre R_sigma y R_eps_rel dentro de la mascara tisular
        loss_ept_sigma = torch.sum(
            F.smooth_l1_loss(r_sigma, torch.zeros_like(r_sigma), reduction="none", beta=0.25)
        ) / tissue_count

        # Escalar el residuo relativo de permitividad a rango O(1) (dividiendo por 50)
        r_eps_scaled = r_eps_rel / 50.0
        loss_ept_eps = torch.sum(
            F.smooth_l1_loss(r_eps_scaled, torch.zeros_like(r_eps_scaled), reduction="none", beta=0.5)
        ) / tissue_count

        # Regularizacion de bordes anatomicos + Huber-TV suave (peso 1e-3) para suprimir oscilaciones de frontera
        sigma_2d = model_outputs["sigma_2d"]
        eps_r_2d = model_outputs["eps_r_2d"]
        m_mag_2d = model_outputs["m_mag_2d"]
        tw_2d = tissue_weight.reshape_as(sigma_2d)
        bg_mask = 1.0 - tw_2d

        loss_huber_tv = 1e-3 * (
            self._huber_tv_smoothness(sigma_2d, delta=0.04)
            + 0.02 * self._huber_tv_smoothness(eps_r_2d, delta=1.5)
        )
        loss_ept_tv = (
            self._edge_aware_smoothness(sigma_2d, m_mag_2d)
            + 1e-4 * self._edge_aware_smoothness(eps_r_2d, m_mag_2d)
            + loss_huber_tv
        )
        loss_ept_bg = torch.mean(bg_mask * ((sigma_2d - 0.02) ** 2 + 1e-4 * (eps_r_2d - 1.0) ** 2))

        loss_helmholtz = (
            eff_lambda_sigma * (loss_ept_sigma + 0.05 * loss_ept_tv + 0.25 * loss_ept_bg)
            + eff_lambda_eps * loss_ept_eps
        )

        total_loss = (
            self.lambda_kspace * loss_kspace
            + self.lambda_coil_smooth * loss_coil
            + self.lambda_harmonic_b0 * loss_harmonic
            + loss_helmholtz
        )

        return {
            "total_loss": total_loss,
            "loss_kspace": loss_kspace.detach(),
            "loss_helmholtz": (loss_ept_sigma + loss_ept_eps).detach(),
            "loss_ept_sigma": loss_ept_sigma.detach(),
            "loss_ept_eps": loss_ept_eps.detach(),
            "loss_coil": loss_coil.detach(),
            "loss_harmonic": loss_harmonic.detach(),
            "ept_active": torch.tensor(1.0 if epoch >= self.warmup_epochs else 0.0, device=total_loss.device),
        }
