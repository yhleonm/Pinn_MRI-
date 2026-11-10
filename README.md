# Decoupled MRI-PINN: Electromagnetic $k$-Space Reconstruction, Electrical Properties Tomography, and Specific Absorption Rate Estimation via Physics-Informed Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20WebSockets-009688.svg)](https://fastapi.tiangolo.com/)
[![React PACS](https://img.shields.io/badge/Frontend-React%2018%20PACS%20Workstation-61dafb.svg)](https://react.dev/)

Plataforma computacional en PyTorch/CUDA y estación interactiva PACS para **reconstrucción electromagnética en espacio-$k$, tomografía de propiedades eléctricas (MR-EPT) y estimación de tasa de absorción específica ($\text{SAR}$)** mediante redes neuronales informadas por la física (PINNs) con fijación de calibre de fase y separación armónica.

---

## 1. Resumen Ejecutivo y Aporte Científico

**Decoupled MRI-PINN** es una arquitectura auto-supervisada específica por corte que resuelve de forma acoplada tres problemas inversos fundamentales de la Física Médica en Resonancia Magnética en **$\sim 3.9\text{ segundos}$** sobre una GPU de consumo (NVIDIA GeForce RTX 4060 Ti):

1. **Reconstrucción Inversa Paralela en Espacio-$k$ Multi-Bobina ($\mathcal{E}$ Differentiable SENSE):** Recupera la magnitud anatómica $|m(\mathbf{x})|$ y la fase compleja libre de solapamiento a partir del **$12.5\%\text{--}25.0\%$** del espacio-$k$ ($R = 4\times$ y $R = 8\times$) con fijación de calibre de fase respecto a una bobina de cuerpo virtual y alineación espectral mediante descomposición en valores singulares (SVD) sobre la región central de autocalibración (ACS).
2. **Tomografía de Propiedades Eléctricas (MR-EPT) Estabilizada en Norma de Sobolev $\mathcal{H}^2(\Omega)$ ($1.5\text{ T}$, $3.0\text{ T}$ y $7.0\text{ T}$):** Evalúa analíticamente el operador Laplaciano espacial de segundo orden ($\nabla^2 B_1^+(\mathbf{x})$) mediante diferenciación automática exacta (`torch.autograd.grad`) sobre una representación neuronal implícita periódica (**SIREN**), acoplada a un desenvolvimiento continuo de fase sobre la variedad $\mathbb{S}^1$ anclado en el anillo cortical periférico y a una proyección ortogonal armónica ($\nabla^2 \phi_{B_0}(\mathbf{x}) = 0$) que suprime los saltos $2\pi$ a $7.0\text{ T}$.
3. **Estimación Cuasi-Estática Local de Tasa de Absorción Específica ($\text{SAR}(\mathbf{x})$ en $\text{W/kg}$):** Estima la distribución espacial de potencia electromagnética absorbida por unidad de masa $\text{SAR}(\mathbf{x}) = \frac{\sigma(\mathbf{x})\|\mathbf{E}(\mathbf{x})\|^2}{2\rho(\mathbf{x})}$ bajo excitación inductiva tipo jaula de pájaro (*birdcage*), permitiendo evaluar incrementos relativos en lesiones hiper-conductivas frente a los umbrales de la norma internacional **IEC 60601-2-33**.

---

## 2. Formulación Físico-Matemática y Arquitectura

```mermaid
flowchart LR
    K["Espacio-k Sub-muestreado<br/>s_c(k) = P(k) F{S_c(x) m(x)}"] --> SVD["Calibre VBC + Alineación SVD<br/>& Escala P99.5 (ACS)"]
    XY["Coordenadas Normalizadas<br/>x_norm in [-1, 1]^2 (requires_grad)"] --> N1["Rama Anatómica SIREN (w0=30)<br/>|m(x)|, phi_tr(x), phi_B0(x)"]
    XY --> N2["Rama Bobinas SIREN (w0=5)<br/>S_c(x), sum |S_c|^2 = 1"]
    XY --> N3["Rama Dieléctrica EPT (w0=15)<br/>sigma(x) [S/m], eps_r(x)"]
    N1 & N2 --> SENSE["Operador Forward SENSE<br/>F_2D{S_c(x) m(x)}"]
    SENSE & SVD --> LK["Pérdida Espectral Espacio-k<br/>||E m - s_k||_2^2"]
    N1 & N3 --> HELM["Operador PDE Helmholtz 2do Orden<br/>autograd.grad + Sobolev H^2(Omega)"]
    HELM --> SAR["Estimación Dosimétrica RF<br/>SAR(x) = sigma |E|^2 / 2 rho [W/kg]"]
```

### Salvaguardas Físico-Numéricas Implementadas
- **Escalado métrico exacto en el dominio de Sitzmann:** Las tres sub-redes SIREN operan en coordenadas adimensionales $\tilde{\mathbf{x}} \in [-1, 1]^2$, transformando las derivadas analíticas al dominio físico métrico $(L_x, L_y)$ mediante la regla de la cadena:
  $$\frac{\partial u}{\partial x_{\text{m}}} = \left(\frac{2}{L_x}\right)\frac{\partial u}{\partial \tilde{x}}, \qquad \frac{\partial^2 u}{\partial x_{\text{m}}^2} = \left(\frac{2}{L_x}\right)^2\frac{\partial^2 u}{\partial \tilde{x}^2}$$
- **Estabilidad numérica en regiones de aire y bordes tisulares:** Denominador regularizado $|B_1^+(\mathbf{x})| + \epsilon_d$ ($\epsilon_d = 10^{-3}$), máscara tisular diferenciable $W_{\text{tis}}(\mathbf{x}) = \operatorname{sigmoid}\!\left(\frac{|m(\mathbf{x})| - 0.15\max|m|}{0.02}\right)$ y regularización de curvatura de fase en norma de Sobolev $\mathcal{H}^2(\Omega)$ en [`src/physics/helmholtz_ept.py`](src/physics/helmholtz_ept.py) y [`src/training/losses.py`](src/training/losses.py).
- **Esquema de calentamiento gradual (*warm-up*) en dos etapas:** Durante las primeras $15$ épocas ($\alpha(t) = 0$) se estabilizan la anatomía $|m(\mathbf{x})|$, la fase $\phi(\mathbf{x})$ y las sensibilidades $S_c(\mathbf{x})$ antes de activar los residuos de segundo orden de Helmholtz ($\alpha(t) \to 1$).

---

## 3. Resultados Experimentales ($N=5$ Semillas) y Figuras a 300 DPI

### 3.1 Evaluación Multi-Campo ($1.5\text{ T}$, $3.0\text{ T}$, $7.0\text{ T}$) con Dispersión de Cole-Cole y Comparación frente a Métodos Clásicos

![Reconstrucción Multi-Campo y Estimación SAR](results/paper_artifacts/fig1_multifield_ept_sar_300dpi.png)

| Estudio / Método Evaluado | Campo $B_0$ ($f_0$) | $\text{PSNR}_{\text{ZF}}$ (dB) | $\text{PSNR}_{\text{CG}}$ (dB) | $\text{PSNR}_{\text{PINN}}$ (dB) | $\text{SSIM}$ | $\text{NRMSE}_{\sigma}$ | $\hat{\sigma}_{\text{ROI}}/\sigma_{\text{GT}}$ (S/m) | $\text{NRMSE}_{\varepsilon_r}$ | $\text{SAR}_{\text{sano}}/\text{SAR}_{\text{tumor}}$ (W/kg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fantoma Numérico ($R=4\times$ Radial, $N=5$)** | $1.5\text{ T}$ ($63.9\text{ MHz}$) | $29.17$ | $36.29$ | **$40.70 \pm 0.38$** | **$0.9572 \pm 0.0130$** | $0.3872 \pm 0.0213$ | $0.94 \pm 0.03 / 1.11$ | $0.3340 \pm 0.0049$ | $0.16$ / **$0.57$** |
| **Fantoma Numérico ($R=4\times$ Radial, $N=5$)** | $3.0\text{ T}$ ($127.7\text{ MHz}$) | $28.97$ | $36.06$ | **$40.51 \pm 1.07$** | $0.9532 \pm 0.0115$ | $0.2956 \pm 0.0138$ | $1.01 \pm 0.05 / 1.20$ | $0.2444 \pm 0.0074$ | $0.62$ / **$2.46$** |
| **Fantoma Numérico ($R=4\times$ Radial, $N=5$)** | $7.0\text{ T}$ ($298.1\text{ MHz}$) | $27.28$ | $34.34$ | **$39.33 \pm 1.26$** | $0.9261 \pm 0.0138$ | **$0.2523 \pm 0.0030$** | **$1.15 \pm 0.05 / 1.32$** | **$0.1649 \pm 0.0091$** | $3.28$ / **$15.03$** |
| **Ref. 1: Adjunto $\mathcal{E}^H s_k$ + SG-EPT ($7\times 7$)** | $3.0\text{ T}$ ($R=4\times$ Pois.) | $30.09$ | $37.46$ | $30.09$ | $0.5712$ | $1.2255$ | $0.99 / 1.20$ | $1.1116$ | $0.96$ / $2.18$ |
| **Ref. 2: CG-SENSE + Savitzky-Golay EPT** | $3.0\text{ T}$ ($R=4\times$ Pois.) | $30.09$ | $37.46$ | $37.46$ | $0.7202$ | $0.7807$ | $1.01 / 1.20$ | $1.1113$ | $0.90$ / $2.38$ |
| **Ref. 3: CG-SENSE + cr-MREPT** | $3.0\text{ T}$ ($R=4\times$ Pois.) | $30.09$ | $37.46$ | $37.46$ | $0.7202$ | $0.6886$ | $0.97 / 1.20$ | $1.1113$ | $0.80$ / $2.26$ |
| **Ablación 1: Imagen PINN ($\lambda_\sigma=0$) + SG-EPT** | $3.0\text{ T}$ ($R=4\times$ Pois.) | $30.09$ | $37.46$ | $41.71$ | $0.9592$ | $0.6220$ | $0.94 / 1.20$ | $1.1163$ | $0.88$ / $2.15$ |
| **Ablación 2: PINN sin desenv. $\mathbb{S}^1$ ($2\pi$)** | $7.0\text{ T}$ ($R=4\times$ Pois.) | $28.76$ | $36.35$ | $31.88$ | $0.9383$ | $0.9429$ | $0.44 / 1.32$ | $0.5689$ | $7.40$ / $5.95$ |
| **Propuesta: Decoupled MRI-PINN Completa** | $3.0\text{ T}$ ($R=4\times$ Pois.) | $30.09$ | $37.46$ | **$41.73$** | **$0.9592$** | **$0.2482$** | **$0.99 / 1.20$** | **$0.2235$** | $0.62$ / **$2.34$** |
| **Propuesta: Decoupled MRI-PINN Completa** | $7.0\text{ T}$ ($R=4\times$ Pois.) | $28.76$ | $36.35$ | **$40.03$** | **$0.9520$** | **$0.2400$** | **$1.16 / 1.32$** | **$0.1569$** | $3.38$ / **$15.54$** |

---

### 3.2 Sensibilidad a la Trayectoria en Espacio-$k$ con Fracción de Muestreo Emparejada ($R = 4\times$ vs $R = 8\times$ a $3.0\text{ T}$)

![Prueba de Estrés R=8x con Fracción Emparejada](results/paper_artifacts/fig2_ablation_8x_trajectories_300dpi.png)

| Aceleración | Trayectoria $\mathbf{P}(\mathbf{k})$ | Muestreo | $\text{PSNR}_{\text{ZF}}$ (dB) | $\text{PSNR}_{\text{CG}}$ (dB) | $\text{PSNR}_{\text{PINN}}$ (dB) | $\text{SSIM}_{\text{PINN}}$ | $\text{NRMSE}_{\sigma}$ | $\text{NRMSE}_{\varepsilon_r}$ | $\hat{\sigma}_{\text{ROI}}/\sigma_{\text{GT}}$ (S/m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$R=4\times$** | `CARTESIAN` | $25.0\%$ | $27.48$ | $28.79$ | $29.59$ ($+2.11$) | $0.8412$ | $0.5318$ | $0.2794$ | $0.94 / 1.20$ |
| **$R=4\times$** | `RADIAL` | $25.3\%$ | $28.99$ | $36.07$ | $40.82$ (**$+11.83$**) | $0.9463$ | $0.3028$ | $0.2484$ | $0.96 / 1.20$ |
| **$R=4\times$** | `POISSON` | $25.0\%$ | $30.09$ | $37.46$ | **$41.73$** ($+11.64$) | **$0.9592$** | **$0.2482$** | **$0.2235$** | **$0.99 / 1.20$** |
| **$R=8\times$** | `CARTESIAN` | $12.5\%$ | $23.92$ | $24.55$ | $25.16$ ($+1.24$) | $0.7553$ | $0.6038$ | $0.3640$ | $1.14 / 1.20$ |
| **$R=8\times$** | `RADIAL` | $12.8\%$ | $24.32$ | $29.30$ | $32.78$ ($+8.46$) | $0.8576$ | $0.4506$ | $0.3190$ | $0.98 / 1.20$ |
| **$R=8\times$** | `POISSON` | $12.5\%$ | $27.14$ | $33.72$ | **$36.01$** (**$+8.87$**) | **$0.9132$** | **$0.3948$** | **$0.2690$** | $1.03 / 1.20$ |

---

### 3.3 Validación Externa sobre Adquisición Cerebral Pública de 8 Bobinas (UC Berkeley / BART ISMRM)

Evaluación con hiperparámetros fijos sobre el corte axial cerebral humano de $8$ bobinas del repositorio público de la Universidad de California en Berkeley (`mrirecon/bart-workshop/ismrm2016/pics-brain`):

![Validación Externa UC Berkeley](results/paper_artifacts/fig3_external_blind_validation_300dpi.png)

| Régimen de Muestreo (UC Berkeley 8-Ch) | Muestreo | $\text{PSNR}_{\text{ZF}}$ (dB) | $\text{PSNR}_{\text{CG}}$ (dB) | $\text{PSNR}_{\text{PINN}}$ (dB) | $\text{SSIM}_{\text{PINN}}$ | $\bar{\sigma}_{\text{par}}$ (S/m) | $\bar{\sigma}_{\text{LCR}}$ (S/m) | $\bar{\varepsilon}_{r,\text{par}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrospectivo Híbrido ($R=4\times$ Radial Áureo)** | $25.3\%$ | $23.21$ | **$31.60$** | $30.46$ ($+7.25$) | $0.8625$ | $0.47$ | $0.90$ | $39.2$ |
| **Retrospectivo Híbrido ($R=4\times$ Poisson-disc)** | $25.0\%$ | $24.10$ | **$34.44$** | $32.38$ ($+8.28$) | $0.8759$ | $0.46$ | $0.90$ | $37.5$ |
| **Prospectivo Crudo Nativo (`ksp.cfl`, $R\approx 8\times$)** | **$13.6\%$** | $21.68$ | $29.11$ | **$29.40$** (**$+7.72$**) | **$0.8344$** | $0.50$ | $0.90$ | $41.0$ |

---

## 4. Estructura del Repositorio

```text
Pinn_MRI-/
├── src/
│   ├── data/
│   │   ├── kspace_loader.py           # Fantomas dispersivos Cole-Cole, controles de confusión y lector HDF5 (.h5)
│   │   └── masks.py                   # Máscaras Cartesianas, Radiales Golden-Angle y Poisson-disc con fracción 1/R exacta
│   ├── physics/
│   │   ├── sense_operator.py          # Operador DifferentiableSENSE (FFT2c/IFFT2c) + calibre VBC y alineación SVD ACS
│   │   └── helmholtz_ept.py           # Operador PDE Helmholtz 2do orden + Estimación Local SAR Birdcage (W/kg)
│   ├── models/decoupled_mri_pinn.py   # Red SIREN desacoplada de 3 ramas (Anatomía, Bobinas, EPT)
│   └── training/
│       ├── losses.py                  # Funcional variacional con norma Sobolev H^2(Omega) y Warm-up físico
│       └── trainer.py                 # Optimización GPU, desenvolvimiento S^1, LBV, CG-SENSE, SG-EPT y cr-MREPT
├── paper/
│   ├── main_raccefyn.tex              # Manuscrito científico en formato RACCEFYN
│   ├── references.bib                 # Bibliografía científica verificada (APA 7 + DOI/PMID)
│   └── main_raccefyn.pdf              # Manuscrito compilado listo para envío
├── backend/                           # Servidor FastAPI + WebSocket (/ws/train) con streaming binario Float32
├── frontend/                          # Estación radiológica PACS en React 18 + TypeScript + Tailwind
├── results/paper_artifacts/           # Figuras blancas de publicación a 300 DPI y métricas JSON
├── train_experiment.py                # CLI de entrenamiento rápido y verificación de gradientes en GPU
├── run_ablation_study.py              # Suite multi-semilla (N=5), comparaciones clásicas y ablación
├── download_and_validate_external.py  # Descarga automática y validación externa (UC Berkeley / BART)
└── start_workstation.ps1              # Lanzador en 1 clic de la estación PACS (Backend + Frontend)
```

---

## 5. Instalación y Reproducción en 1 Clic

### Requisitos
- Python $3.10+$ con PyTorch (CUDA recomendado)
- Node.js $18+$ (para la interfaz interactiva PACS)

```bash
pip install -r requirements.txt
```

### A. Ejecutar validación rápida en terminal
```bash
python train_experiment.py --mode synthetic --epochs 50
```

### B. Reproducir todas las tablas $\LaTeX$ y figuras a 300 DPI del artículo
```bash
# 1. Estudios 1, 2 y 3 (Multi-campo 1.5T/3.0T/7.0T, R=4x/8x, controles y ablación N=5)
python run_ablation_study.py

# 2. Estudio 4 (Descarga automática y Validación Externa sobre UC Berkeley 8-Ch Brain)
python download_and_validate_external.py
```

### C. Iniciar la Estación Interactiva PACS (Backend FastAPI + Visor React)
En Windows PowerShell:
```powershell
.\start_workstation.ps1
```
O manualmente en dos terminales:
```bash
# Terminal 1 (Backend GPU en puerto 8000)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 (Frontend PACS en puerto 5173)
cd frontend && npm install && npm run dev
```

Dentro del visor PACS (`http://localhost:5173`):
- Puedes alternar entre el fantoma cortical (`data/cortical_brain_tumor_8coils_3T.h5`), el fantoma numérico con ruido correlacionado (`data/numerical_brain_phantom_8coils.h5`) o la adquisición externa real de UC Berkeley (`data/external_public_brain_raw.h5`).
- En la barra superior derecha del visor puedes activar **`Mostrar Contorno ROI Tumor`** y **`Dosimetría SAR (W/kg)`** en tiempo real.

---

## 6. Citación

Si este código o formulación matemática resulta útil en tu investigación, por favor cita nuestro trabajo (ver [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{leon2026decoupledmripinn,
  author  = {Le{\'o}n-Monta{\~n}ez, Yerson Hernando},
  title   = {Reconstrucci{\'o}n electromagn{\'e}tica en espacio-k, tomograf{\'i}a de propiedades el{\'e}ctricas y estimaci{\'o}n de tasa de absorci{\'o}n espec{\'i}fica mediante redes neuronales informadas por la f{\'i}sica},
  journal = {Revista de la Academia Colombiana de Ciencias Exactas, F{\'i}sicas y Naturales (RACCEFYN)},
  year    = {2026},
  url     = {https://github.com/yhleonm/Pinn_MRI-}
}
```
