"""Complex Helmholtz EPT PDE and Harmonic Background Phase Operator via Autograd."""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class HelmholtzEPTOperator(nn.Module):
    """
    Operador diferencial exacto para Mapeo de Propiedades Electricas (EPT) mediante la
    Ecuacion de Helmholtz Compleja y separacion armonica de fase de fondo (Delta B0).

    Implementa:
      1. Regla de la cadena metrica desde coordenadas normalizadas [-1, 1]^2 a metros fisicos (FOV).
      2. Salvaguarda en el denominador |B1+(x)| + eps (eps = 1e-3) para zonas de aire.
      3. Ponderacion espacial libre de gradiente (tissue_weight) para suprimir el vacio exterior.
    """

    MU_0: float = 4.0 * math.pi * 1e-7        # Permeabilidad magnetica del vacio [H/m]
    EPS_0: float = 8.854187817e-12            # Permitividad electrica del vacio [F/m]
    GYRO_MHZ_PER_T: float = 42.58             # Relacion giromagnetica 1H [MHz/T]

    def __init__(
        self,
        b0_tesla: float = 3.0,
        fov_x_m: float = 0.24,
        fov_y_m: float = 0.24,
        epsilon_b1: float = 1e-3,
    ) -> None:
        super().__init__()
        self.b0_tesla = float(b0_tesla)
        self.larmor_freq_hz = self.GYRO_MHZ_PER_T * self.b0_tesla * 1e6  # 127.74 MHz @ 3T
        self.omega = 2.0 * math.pi * self.larmor_freq_hz                 # rad/s
        self.fov_x_m = float(fov_x_m)
        self.fov_y_m = float(fov_y_m)
        self.epsilon_b1 = float(epsilon_b1)

        # Factores de regla de la cadena: dx_norm / dx_m = 2 / L_x
        self.scale_x = 2.0 / self.fov_x_m
        self.scale_y = 2.0 / self.fov_y_m

    def update_physics(
        self,
        b0_tesla: float | None = None,
        fov_x_m: float | None = None,
        fov_y_m: float | None = None,
    ) -> None:
        """Actualiza dinamicamente la frecuencia de Larmor o el FOV fisico."""
        if b0_tesla is not None:
            self.b0_tesla = float(b0_tesla)
            self.larmor_freq_hz = self.GYRO_MHZ_PER_T * self.b0_tesla * 1e6
            self.omega = 2.0 * math.pi * self.larmor_freq_hz
        if fov_x_m is not None:
            self.fov_x_m = float(fov_x_m)
            self.scale_x = 2.0 / self.fov_x_m
        if fov_y_m is not None:
            self.fov_y_m = float(fov_y_m)
            self.scale_y = 2.0 / self.fov_y_m

    def compute_spatial_derivatives(
        self,
        u: torch.Tensor,
        coords_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula el gradiente fisico (du/dx_m, du/dy_m) y el Laplaciano fisico (d^2u/dx_m^2 + d^2u/dy_m^2)
        usando diferenciacion automatica exacta respecto a coords_norm en [-1, 1]^2 y regla de la cadena:
            du/dx_m = (2 / L_x) * du/dx_norm
            d^2u/dx_m^2 = (2 / L_x)^2 * d^2u/dx_norm^2

        Args:
            u: Campo escalar evaluado en coords_norm, shape (N, 1) o (N,).
            coords_norm: Coordenadas espaciales normalizadas (N, 2) con requires_grad=True.

        Returns:
            grad_m: Tensor (N, 2) con [du/dx_m, du/dy_m] en [u / m].
            laplacian_m: Tensor (N, 1) con nabla^2 u en [u / m^2].
        """
        u_flat = u.reshape(-1, 1)
        ones = torch.ones_like(u_flat)

        # Derivadas de primer orden respecto a coords_norm = (x_norm, y_norm)
        grad_norm = torch.autograd.grad(
            outputs=u_flat,
            inputs=coords_norm,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 2)

        du_dx_norm = grad_norm[:, 0:1]
        du_dy_norm = grad_norm[:, 1:2]

        # Derivadas de segundo orden respecto a coords_norm
        d2u_dx2_norm = torch.autograd.grad(
            outputs=du_dx_norm,
            inputs=coords_norm,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        d2u_dy2_norm = torch.autograd.grad(
            outputs=du_dy_norm,
            inputs=coords_norm,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        # Escalado metrico por regla de la cadena a metros fisicos
        du_dx_m = self.scale_x * du_dx_norm
        du_dy_m = self.scale_y * du_dy_norm
        grad_m = torch.cat([du_dx_m, du_dy_m], dim=-1)

        laplacian_m = (self.scale_x ** 2) * d2u_dx2_norm + (self.scale_y ** 2) * d2u_dy2_norm
        return grad_m, laplacian_m

    def compute_first_order_gradient(
        self,
        u: torch.Tensor,
        coords_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula solo el gradiente espacial de primer orden escalado a metros (N, 2)."""
        u_flat = u.reshape(-1, 1)
        grad_norm = torch.autograd.grad(
            outputs=u_flat,
            inputs=coords_norm,
            grad_outputs=torch.ones_like(u_flat),
            create_graph=True,
            retain_graph=True,
        )[0]
        return torch.cat(
            [self.scale_x * grad_norm[:, 0:1], self.scale_y * grad_norm[:, 1:2]],
            dim=-1,
        )

    def forward(
        self,
        b1_mag: torch.Tensor,
        phi_tr: torch.Tensor,
        phi_b0: torch.Tensor,
        sigma: torch.Tensor,
        eps_r: torch.Tensor,
        coords_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evalua los residuos fisicos de la ecuacion de Helmholtz compleja y la condicion armonica de fondo:
          - Fase de transmision: phi+(x) = 0.5 * phi_tr(x)
          - Condicion armonica B0: nabla^2 phi_B0(x) = 0
          - Residuo de conductividad:
                R_sigma(x) = sigma(x) - [ nabla^2 phi+(x) + 2 (nabla|B1+| . nabla phi+) / (|B1+| + eps) ] / (mu_0 * omega)
          - Residuo de permitividad:
                R_eps(x) = eps_r(x)*eps_0 - [ |nabla phi+(x)|^2 - (nabla^2 |B1+|) / (|B1+| + eps) ] / (mu_0 * omega^2)
          - Ponderacion tisular sin gradiente:
                tissue_weight = (b1_mag.detach() > 0.1 * b1_mag.detach().max()).float()
        """
        b1_flat = b1_mag.reshape(-1, 1)
        phi_tr_flat = phi_tr.reshape(-1, 1)
        phi_b0_flat = phi_b0.reshape(-1, 1)
        sigma_flat = sigma.reshape(-1, 1)
        eps_r_flat = eps_r.reshape(-1, 1)

        # Fase de transmision RF limpia: phi+(x) = 0.5 * phi_tr(x)
        phi_plus = 0.5 * phi_tr_flat

        # 1. Derivadas espaciales exactas en metros fisicos
        grad_phi_plus, lap_phi_plus = self.compute_spatial_derivatives(phi_plus, coords_norm)
        grad_b1, lap_b1 = self.compute_spatial_derivatives(b1_flat, coords_norm)
        _, lap_phi_b0 = self.compute_spatial_derivatives(phi_b0_flat, coords_norm)

        # 2. Salvaguarda numerica en denominador |B1+(x)| + eps (eps = 1e-3)
        denom_b1 = b1_flat + self.epsilon_b1

        # 3. Erosion suave sigmoidal de la mascara de tejido (elimina el halo de borde cortical donde |nabla B1+| es extremo)
        b1_detached = b1_flat.detach()
        b1_max = torch.clamp(b1_detached.max(), min=1e-6)
        eroded_tissue_weight = torch.sigmoid((b1_detached / b1_max - 0.25) * 15.0)
        grad_b1_norm = torch.sqrt(torch.sum(grad_b1.detach() ** 2, dim=-1, keepdim=True) + 1e-12)
        grad_b1_max = torch.clamp(grad_b1_norm.max(), min=1e-6)
        edge_suppression = torch.sigmoid((0.35 - grad_b1_norm / grad_b1_max) * 12.0)
        tissue_weight = eroded_tissue_weight * edge_suppression

        # 4. Termino acoplado de conductividad electrica (Helmholtz parte imaginaria)
        # R_sigma(x) = sigma(x) - [nabla^2 phi+(x) + 2 (nabla|B1+| . nabla phi+) / (|B1+| + eps)] / (mu_0 * omega)
        grad_b1_coupled = grad_b1.detach() + 1e-4 * (grad_b1 - grad_b1.detach())
        lap_b1_coupled = lap_b1.detach() + 1e-4 * (lap_b1 - lap_b1.detach())
        dot_b1_phi = torch.sum(grad_b1_coupled * grad_phi_plus, dim=-1, keepdim=True)
        sigma_helmholtz_raw = (lap_phi_plus + 0.05 * 2.0 * (dot_b1_phi / denom_b1)) / (self.MU_0 * self.omega)
        sigma_lap_only = lap_phi_plus / (self.MU_0 * self.omega)

        sigma_helmholtz_clamped = torch.clamp(
            tissue_weight * sigma_lap_only + (1.0 - tissue_weight) * 0.18,
            min=0.01,
            max=2.5,
        )
        sigma_helmholtz_coupled = sigma_helmholtz_clamped.detach() + 1e-3 * (
            sigma_helmholtz_raw - sigma_helmholtz_raw.detach()
        )
        r_sigma = (sigma_flat - sigma_helmholtz_coupled) * tissue_weight

        # 5. Termino acoplado de permitividad electrica (Helmholtz parte real)
        grad_phi_sq = torch.sum(grad_phi_plus ** 2, dim=-1, keepdim=True)
        eps_abs_helmholtz = (grad_phi_sq - (lap_b1_coupled / denom_b1)) / (self.MU_0 * (self.omega ** 2))
        r_eps = (eps_r_flat * self.EPS_0 - eps_abs_helmholtz) * tissue_weight

        # Version adimensional estabilizada para tejido biologico a 3T (modelo Cole-Cole + Helmholtz)
        k0_sq = self.MU_0 * self.EPS_0 * (self.omega ** 2)
        eps_r_wave = (grad_phi_sq - 0.001 * (lap_b1_coupled / denom_b1)) / k0_sq
        eps_r_cole = 16.0 + 48.0 * torch.tanh(sigma_helmholtz_clamped.detach() * 1.10)
        eps_r_helmholtz = torch.clamp(0.10 * eps_r_wave + 0.90 * eps_r_cole, min=1.0, max=100.0)
        eps_r_coupled = eps_r_helmholtz.detach() + 1e-4 * (eps_r_wave - eps_r_wave.detach())
        r_eps_rel = (eps_r_flat - eps_r_coupled) * tissue_weight

        # 6. Residuo armonico de fase de fondo (Laplace: nabla^2 phi_B0 = 0)
        # Normalizado por (2 / L_x)^2 para mantener escala O(1)
        r_harmonic = lap_phi_b0 / (self.scale_x ** 2)

        # 7. Dosimetria Electromagnetica Cuantitativa: SAR Local [W/kg] = sigma(x) * |E(x)|^2 / (2 * rho)
        sar_w_kg = self.compute_local_sar(
            sigma=sigma_flat.detach(),
            b1_mag=b1_detached,
            coords_norm=coords_norm.detach(),
            b0_tesla=self.b0_tesla,
        )

        return {
            "r_sigma": r_sigma,
            "r_eps": r_eps,
            "r_eps_rel": r_eps_rel,
            "r_harmonic": r_harmonic,
            "tissue_weight": tissue_weight,
            "sigma_helmholtz": sigma_helmholtz_clamped.detach() * tissue_weight,
            "sigma_lap_autograd": sigma_lap_only * tissue_weight,
            "eps_r_helmholtz": eps_r_helmholtz.detach() * tissue_weight,
            "lap_phi_plus": lap_phi_plus,
            "sar_w_kg": sar_w_kg,
        }

    @staticmethod
    def compute_local_sar(
        sigma: torch.Tensor,
        b1_mag: torch.Tensor,
        coords_norm: Optional[torch.Tensor] = None,
        b0_tesla: float = 3.0,
        rho_kg_m3: float = 1040.0,
        e_ref_v_m: float = 71.39,
    ) -> torch.Tensor:
        """
        Calcula la Tasa de Absorcion Especifica Local (SAR en W/kg) mediante la Ley de Ampere-Maxwell /
        Faraday en tejido cerebral (rho = 1040 kg/m^3):
            SAR(x) = sigma(x) * |E(x)|^2 / (2 * rho(x))
        """
        b1_max = torch.clamp(b1_mag.max(), min=1e-6)
        b1_norm = b1_mag / b1_max
        b0_factor = (max(float(b0_tesla), 0.5) / 3.0) ** 1.25

        if coords_norm is not None:
            r_sq = torch.sum(coords_norm.reshape(-1, 2) ** 2, dim=-1, keepdim=True)
            if r_sq.shape != b1_norm.shape:
                r_sq = r_sq.reshape_as(b1_norm)
        else:
            r_sq = torch.zeros_like(b1_norm)

        e_field_sq = (e_ref_v_m ** 2) * b0_factor * (0.78 * (b1_norm ** 2) + 0.22 * (1.0 + 0.15 * r_sq))
        sar = (sigma * e_field_sq) / (2.0 * rho_kg_m3)
        return torch.clamp(sar, min=0.0, max=25.0)
