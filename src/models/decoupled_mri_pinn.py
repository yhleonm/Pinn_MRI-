"""Decoupled 3-Branch MRI-PINN Architecture for Joint SENSE Reconstruction and Helmholtz EPT."""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .siren import SIRENNetwork


def bounded_softplus(
    x: torch.Tensor,
    low: float,
    high: float,
    center_offset: float = 0.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Activacion Softplus doblemente acotada C^infinito en el intervalo [low, high]:
        f(z) = low + softplus(z) - softplus(z - (high - low))
    """
    z = x + center_offset
    span = high - low
    return low + F.softplus(z, beta=beta) - F.softplus(z - span, beta=beta)


class AnatomyAndPhaseSIREN(nn.Module):
    """
    Rama 1: Red SIREN que predice:
      - |m(x)|: Magnitud espinorial / anatomica no negativa (omega_0 = 30 para bordes nitidos).
      - phi_tr(x): Fase transceptora de radiofrecuencia (2 * phi+, C^2 diferenciable para Laplaciano).
      - phi_B0(x): Fase parasita armonica de fondo (Delta B0).
    """

    def __init__(
        self,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        # Red SIREN de alta frecuencia (omega_0 = 30) para magnitud anatomica |m(x)|
        self.mag_net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=1,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            final_scale=0.5,
        )
        # Red SIREN C^2 en espacio de Sobolev H^2 para fase transceptora RF phi_tr(x)
        self.phase_tr_net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=3,
            out_features=1,
            first_omega_0=5.0,
            hidden_omega_0=5.0,
            final_scale=0.15,
        )
        # Sub-red armonica de baja frecuencia (omega_0 = 2.5) para fase parasita phi_B0(x)
        self.b0_net = SIRENNetwork(
            in_features=2,
            hidden_features=32,
            hidden_layers=2,
            out_features=1,
            first_omega_0=2.5,
            hidden_omega_0=2.5,
            final_scale=0.15,
        )

    def forward(
        self, coords_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_mag = self.mag_net(coords_norm)
        phi_tr = self.phase_tr_net(coords_norm)

        # Magnitud positiva con supresion limpia de fondo de aire
        m_mag = F.softplus(raw_mag - 0.35, beta=6.0)
        phi_b0 = self.b0_net(coords_norm)
        phi_total = phi_tr + phi_b0

        return m_mag, phi_tr, phi_b0, phi_total


class CoilSensitivityNet(nn.Module):
    """
    Rama 2: Red SIREN suave de baja frecuencia (omega_0 = 5) que predice las partes
    real e imaginaria de los Nc mapas de sensibilidad C_j(x) con restriccion exacta
    de suma de cuadrados (SoS): sum_{j=1}^{Nc} |C_j(x)|^2 = 1.
    """

    def __init__(
        self,
        num_coils: int = 8,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        omega_0: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_coils = int(num_coils)
        self.net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=2 * self.num_coils,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            final_scale=1.0,
        )

    def forward(
        self, coords_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna:
            coil_complex: Tensor complejo (N, Nc) con sum_j |C_j(x)|^2 = 1.
            coil_real: Parte real normalizada (N, Nc).
            coil_imag: Parte imaginaria normalizada (N, Nc).
        """
        raw = self.net(coords_norm)  # (N, 2 * Nc)
        raw_real = raw[..., : self.num_coils]
        raw_imag = raw[..., self.num_coils :]

        # Normalizacion estricta Suma de Cuadrados (SoS = 1) y fijacion de gauge de fase comun
        sos = torch.sqrt(
            torch.sum(raw_real ** 2 + raw_imag ** 2, dim=-1, keepdim=True) + 1e-8
        )
        coil_real = raw_real / sos
        coil_imag = raw_imag / sos
        coil_complex_raw = torch.complex(coil_real, coil_imag)

        common_sum = torch.sum(coil_complex_raw, dim=-1, keepdim=True)
        common_phase = torch.angle(common_sum)
        demod = torch.polar(torch.ones_like(common_phase), -common_phase)
        coil_complex = coil_complex_raw * demod
        return coil_complex, coil_complex.real, coil_complex.imag


class ElectricalPropertiesNet(nn.Module):
    """
    Rama 3: Cabeza fisica que parametriza los mapas cuantitativos de:
      - Conductividad electrica: sigma(x) in [0.01, 3.0] S/m (Softplus acotada).
      - Permitividad relativa: eps_r(x) in [1.0, 100.0] adimensional (Softplus acotada).
    """

    def __init__(
        self,
        hidden_features: int = 96,
        hidden_layers: int = 3,
        omega_0: float = 15.0,
        sigma_bounds: Tuple[float, float] = (0.01, 3.0),
        eps_r_bounds: Tuple[float, float] = (1.0, 100.0),
    ) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_bounds[0]), float(sigma_bounds[1])
        self.eps_min, self.eps_max = float(eps_r_bounds[0]), float(eps_r_bounds[1])

        self.net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=2,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            final_scale=0.5,
        )

    def forward(self, coords_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(coords_norm)
        raw_sigma = out[..., 0:1]
        raw_eps = out[..., 1:2]

        # Conductividad acotada en [0.01, 3.0] S/m con inicializacion centrada en ~0.55 S/m
        sigma = bounded_softplus(
            raw_sigma,
            low=self.sigma_min,
            high=self.sigma_max,
            center_offset=0.25,
            beta=2.0,
        )
        # Permitividad relativa acotada en [1.0, 100.0] con inicializacion centrada en ~50
        eps_r = bounded_softplus(
            raw_eps * 25.0,
            low=self.eps_min,
            high=self.eps_max,
            center_offset=50.0,
            beta=0.2,
        )
        return sigma, eps_r


class DecoupledMRIPINN(nn.Module):
    """
    Arquitectura Principal Desacoplada de 3 Ramas para Reconstruccion MRI SENSE Libre de
    Calibracion e Inversion Cuantitativa de Propiedades Electricas (Helmholtz EPT).

    Entrada estricta: coords_norm in [-1, 1]^2 con requires_grad_(True).
    """

    def __init__(
        self,
        num_coils: int = 8,
        omega_anatomy: float = 30.0,
        omega_coil: float = 5.0,
        omega_ept: float = 15.0,
        hidden_features_anatomy: int = 128,
        hidden_layers_anatomy: int = 4,
        hidden_features_coil: int = 64,
        hidden_layers_coil: int = 3,
        hidden_features_ept: int = 96,
        hidden_layers_ept: int = 3,
        sigma_bounds: Tuple[float, float] = (0.01, 3.0),
        eps_r_bounds: Tuple[float, float] = (1.0, 100.0),
    ) -> None:
        super().__init__()
        self.num_coils = int(num_coils)

        self.anatomy_branch = AnatomyAndPhaseSIREN(
            hidden_features=hidden_features_anatomy,
            hidden_layers=hidden_layers_anatomy,
            omega_0=omega_anatomy,
        )
        self.coil_branch = CoilSensitivityNet(
            num_coils=num_coils,
            hidden_features=hidden_features_coil,
            hidden_layers=hidden_layers_coil,
            omega_0=omega_coil,
        )
        self.ept_branch = ElectricalPropertiesNet(
            hidden_features=hidden_features_ept,
            hidden_layers=hidden_layers_ept,
            omega_0=omega_ept,
            sigma_bounds=sigma_bounds,
            eps_r_bounds=eps_r_bounds,
        )

    @staticmethod
    def create_normalized_grid(
        height: int,
        width: int,
        device: torch.device | str = "cuda",
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """
        Genera la malla de coordenadas espaciales normalizadas (x_norm, y_norm) in [-1, 1]^2
        con shape (H * W, 2) y requires_grad_(True).
        """
        y_vals = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
        x_vals = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_vals, x_vals, indexing="ij")
        coords_norm = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        if requires_grad:
            coords_norm.requires_grad_(True)
        return coords_norm

    def forward(
        self,
        coords_norm: torch.Tensor,
        grid_shape: Tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evalua las 3 ramas sobre coords_norm (N, 2) in [-1, 1]^2.

        Si se especifica grid_shape = (H, W), incluye tambien las vistas 2D listas para
        el operador DifferentiableSENSE:
          - m_complex_2d: (H, W) [complex64]
          - coil_maps_2d: (Nc, H, W) [complex64]
        """
        m_mag, phi_tr, phi_b0, phi_total = self.anatomy_branch(coords_norm)
        coil_complex, coil_real, coil_imag = self.coil_branch(coords_norm)
        sigma, eps_r = self.ept_branch(coords_norm)

        m_complex = torch.polar(m_mag.squeeze(-1), phi_total.squeeze(-1))  # (N,)

        outputs: Dict[str, torch.Tensor] = {
            "m_mag": m_mag,                 # (N, 1)
            "phi_tr": phi_tr,               # (N, 1)
            "phi_b0": phi_b0,               # (N, 1)
            "phi_total": phi_total,         # (N, 1)
            "m_complex": m_complex,         # (N,)
            "coil_complex": coil_complex,   # (N, Nc)
            "coil_real": coil_real,         # (N, Nc)
            "coil_imag": coil_imag,         # (N, Nc)
            "sigma": sigma,                 # (N, 1)
            "eps_r": eps_r,                 # (N, 1)
        }

        if grid_shape is not None:
            h, w = grid_shape
            outputs["m_complex_2d"] = m_complex.reshape(h, w)
            outputs["coil_maps_2d"] = coil_complex.transpose(0, 1).reshape(self.num_coils, h, w)
            outputs["m_mag_2d"] = m_mag.reshape(h, w)
            outputs["phi_tr_2d"] = phi_tr.reshape(h, w)
            outputs["phi_b0_2d"] = phi_b0.reshape(h, w)
            outputs["sigma_2d"] = sigma.reshape(h, w)
            outputs["eps_r_2d"] = eps_r.reshape(h, w)

        return outputs
