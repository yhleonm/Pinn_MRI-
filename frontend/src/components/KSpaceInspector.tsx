import React, { useEffect, useRef } from "react";
import { Cpu, Layers, Radio } from "lucide-react";
import { DatasetState, DecodedMaps } from "../types";
import { renderMatrixToImageData } from "../utils/colormaps";

interface KSpaceInspectorProps {
  datasetState: DatasetState | null;
  maps: DecodedMaps;
}

export const KSpaceInspector: React.FC<KSpaceInspectorProps> = ({
  datasetState,
  maps,
}) => {
  const fullKRef = useRef<HTMLCanvasElement | null>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const maskedKRef = useRef<HTMLCanvasElement | null>(null);
  const phaseRef = useRef<HTMLCanvasElement | null>(null);

  const width = datasetState?.width || 128;
  const height = datasetState?.height || 128;

  useEffect(() => {
    const drawPanel = (
      canvas: HTMLCanvasElement | null,
      matrix: Float32Array | undefined,
      minV: number,
      maxV: number,
      cmap: "grayscale" | "viridis" | "inferno" | "magma" | "turbo"
    ) => {
      if (!canvas || !matrix || matrix.length !== width * height) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const imgData = renderMatrixToImageData(
        matrix,
        width,
        height,
        minV,
        maxV,
        cmap
      );
      ctx.putImageData(imgData, 0, 0);
    };

    // Calcular rango maximo para el log-espectro
    let maxK = 4.0;
    if (maps.kspace_log_full) {
      let m = 0;
      for (let i = 0; i < maps.kspace_log_full.length; i++) {
        if (maps.kspace_log_full[i] > m) m = maps.kspace_log_full[i];
      }
      if (m > 0.1) maxK = m * 0.85;
    }

    drawPanel(fullKRef.current, maps.kspace_log_full, 0.0, maxK, "inferno");
    drawPanel(maskRef.current, maps.mask, 0.0, 1.0, "grayscale");
    drawPanel(maskedKRef.current, maps.kspace_log_masked, 0.0, maxK, "inferno");
    drawPanel(phaseRef.current, maps.phi_tr, -1.8, 0.4, "turbo");
  }, [maps, width, height]);

  const items = [
    {
      ref: fullKRef,
      title: "Espectro K-Space Completo",
      subtitle: "log(1 + ||s(k)||₂) (8 Bobinas)",
      tag: "FFT2 Ortho",
    },
    {
      ref: maskRef,
      title: `Máscara P(k) (${datasetState?.mask_type?.toUpperCase() || "CARTESIAN"})`,
      subtitle: `Aceleración R=${datasetState?.acceleration || 4}x | ACS=${
        datasetState?.acs_lines || 18
      }`,
      tag: `${((datasetState?.sampling_fraction || 0.25) * 100).toFixed(1)}% Muestreado`,
    },
    {
      ref: maskedKRef,
      title: "K-Space Sub-Muestreado",
      subtitle: "P(k) ⊙ F{C_j(x) · m(x)}",
      tag: `R = ${datasetState?.acceleration || 4}x`,
    },
    {
      ref: phaseRef,
      title: "Fase Transceptora RF φ_tr(x)",
      subtitle: "φ_tr(x) = 2φ⁺(x) Libre de Fondo ΔB₀",
      tag: "Entrada Helmholtz",
    },
  ];

  return (
    <div className="bg-slate-900/70 border border-slate-800/90 rounded-xl p-4 flex flex-col gap-3 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Inspector de Dominio de Frecuencia (K-Space) y Fase Transceptora RF
          </h3>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
          <span className="flex items-center gap-1">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            Fuente: <strong className="text-slate-200">{datasetState?.source || "Synthetic"}</strong>
          </span>
          <span className="flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            VRAM: <strong className="text-emerald-300">{datasetState?.vram_allocated_mb?.toFixed(1) || "0.0"} MB</strong>
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {items.map((item, idx) => (
          <div
            key={idx}
            className="bg-slate-950 border border-slate-800/80 rounded-lg p-2.5 flex items-center gap-3"
          >
            <div className="w-20 h-20 shrink-0 bg-black rounded border border-slate-800 overflow-hidden">
              <canvas
                ref={item.ref}
                width={width}
                height={height}
                style={{ imageRendering: "pixelated" }}
                className="w-full h-full object-contain"
              />
            </div>
            <div className="flex flex-col justify-between min-w-0">
              <div>
                <span className="inline-block text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-300 mb-1">
                  {item.tag}
                </span>
                <h4 className="text-xs font-semibold text-slate-100 truncate">
                  {item.title}
                </h4>
                <p className="text-[11px] text-slate-400 leading-snug">
                  {item.subtitle}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
