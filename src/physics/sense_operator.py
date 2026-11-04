"""Differentiable Multi-Coil SENSE Operator (Forward & Adjoint) in PyTorch."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class DifferentiableSENSE(nn.Module):
    """
    Operador fisico diferenciable Forward (E) y Adjoint (E^H) para MRI multi-bobina.

    Ecuacion Forward:
        s_hat_j(k) = P(k) * F_ortho{ C_j(x) * m(x) }

    Ecuacion Adjoint:
        m_zf(x) = sum_{j=1}^{N_c} conj(C_j(x)) * F_ortho^{-1}{ P(k) * s_j(k) }
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def fft2c(image: torch.Tensor) -> torch.Tensor:
        """Transformada Rapida de Fourier 2D centrada y ortonormal."""
        return torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(image, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ),
            dim=(-2, -1),
        )

    @staticmethod
    def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
        """Transformada Rapida de Fourier Inversa 2D centrada y ortonormal."""
        return torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(kspace, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ),
            dim=(-2, -1),
        )

    def forward(
        self,
        m_complex: torch.Tensor,
        coil_maps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aplica el operador Forward SENSE (E).

        Args:
            m_complex: Imagen compleja en dominio espacial, shape (H, W) o (B, 1, H, W) [complex64].
            coil_maps: Mapas complejos de sensibilidad C_j(x), shape (Nc, H, W) o (B, Nc, H, W) [complex64].
            mask: Mascara binaria de sub-muestreo P(k), shape (H, W) o (1, H, W).

        Returns:
            Espacio-k predicho por bobina s_hat_j(k), shape (Nc, H, W) [complex64].
        """
        if m_complex.ndim == 2:
            m_expanded = m_complex.unsqueeze(0)  # (1, H, W)
        else:
            m_expanded = m_complex

        coil_images = coil_maps * m_expanded  # (Nc, H, W)
        kspace_pred = self.fft2c(coil_images)

        if mask is not None:
            if mask.ndim == 2:
                mask_expanded = mask.unsqueeze(0)
            else:
                mask_expanded = mask
            kspace_pred = kspace_pred * mask_expanded

        return kspace_pred

    def adjoint(
        self,
        masked_kspace: torch.Tensor,
        coil_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aplica el operador Adjoint SENSE (E^H) para reconstruccion Zero-Filled combinada.

        Args:
            masked_kspace: Espacio-k sub-muestreado s_j(k), shape (Nc, H, W) [complex64].
            coil_maps: Mapas complejos de sensibilidad C_j(x), shape (Nc, H, W) [complex64].
                       Si es None, retorna combinacion Root-Sum-of-Squares (RSS) con fase promedio.

        Returns:
            Imagen compleja reconstruida en el dominio espacial, shape (H, W) [complex64].
        """
        coil_images = self.ifft2c(masked_kspace)  # (Nc, H, W)

        if coil_maps is not None:
            return torch.sum(torch.conj(coil_maps) * coil_images, dim=-3)

        # Fallback sin mapas: magnitud RSS + fase ponderada por magnitud
        rss_mag = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=-3) + 1e-12)
        weighted_complex = torch.sum(coil_images * torch.abs(coil_images), dim=-3)
        phase = torch.angle(weighted_complex)
        return torch.polar(rss_mag, phase)

    def estimate_initial_coil_maps(
        self,
        kspace: torch.Tensor,
        acs_width: int = 20,
    ) -> torch.Tensor:
        """
        Estima mapas de sensibilidad suaves de referencia a partir de la region central ACS
        aplicando ventana Hanning 2D en baja frecuencia, normalizacion SoS y preservacion
        de la fase transceptora comun phi(x) = phi_tr(x) + phi_B0(x) en el objeto m(x).
        """
        nc, h, w = kspace.shape
        device = kspace.device

        # Ventana Hanning concentrada en la region central ACS
        win_y = torch.zeros(h, device=device, dtype=torch.float32)
        win_x = torch.zeros(w, device=device, dtype=torch.float32)
        cy, cx = h // 2, w // 2
        ry = min(acs_width, h // 2)
        rx = min(acs_width, w // 2)

        win_y[cy - ry : cy + ry] = torch.hann_window(2 * ry, device=device)
        win_x[cx - rx : cx + rx] = torch.hann_window(2 * rx, device=device)
        window_2d = torch.outer(win_y, win_x).unsqueeze(0)  # (1, H, W)

        lowres_kspace = kspace * window_2d
        lowres_imgs = self.ifft2c(lowres_kspace)  # (Nc, H, W)

        # Alineacion SVD de desfase electronico de bobinas si existen saltos de cable (> 0.8 rad en el centro)
        center_phases = torch.angle(lowres_imgs[:, cy, cx])
        phase_spread = torch.max(
            torch.abs(
                torch.atan2(
                    torch.sin(center_phases - center_phases[0]),
                    torch.cos(center_phases - center_phases[0]),
                )
            )
        )
        if float(phase_spread.item()) > 0.8:
            flat_low = lowres_imgs.reshape(nc, -1)
            cov = torch.matmul(flat_low, torch.conj(flat_low.T))
            _, eigvecs = torch.linalg.eigh(cov)
            dom_vec = eigvecs[:, -1]  # Modo dominante SVD (Nc,)
            cable_phasor = torch.polar(torch.ones(nc, device=device), -torch.angle(dom_vec)).view(
                nc, 1, 1
            )
            lowres_aligned = lowres_imgs * cable_phasor
            common_sum = torch.sum(lowres_aligned, dim=0, keepdim=True)
        else:
            common_sum = torch.sum(lowres_imgs, dim=0, keepdim=True)

        # Fase transceptora comun (modo comun del arreglo de bobinas): sum_j I_low,j(x)
        common_phase = torch.angle(common_sum)
        phase_demod = torch.polar(torch.ones_like(common_phase), -common_phase)

        sos = torch.sqrt(torch.sum(torch.abs(lowres_imgs) ** 2, dim=0, keepdim=True) + 1e-8)
        return (lowres_imgs * phase_demod) / sos
