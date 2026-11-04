import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Crosshair,
  Eye,
  Flame,
  Maximize2,
  RotateCcw,
  Sliders,
  Sparkles,
  Target,
} from "lucide-react";
import {
  ColormapName,
  CursorProbe,
  DatasetState,
  DecodedMaps,
} from "../types";
import {
  getCssGradientForColormap,
  renderMatrixToImageData,
} from "../utils/colormaps";

interface MultiPanelViewerProps {
  datasetState: DatasetState | null;
  maps: DecodedMaps;
}

interface PanelSpec {
  id: string;
  title: string;
  subtitle: string;
  unit: string;
  matrix: Float32Array | undefined;
  minVal: number;
  maxVal: number;
  colormap: ColormapName;
  badgeText: string;
  badgeColor: string;
}

export const MultiPanelViewer: React.FC<MultiPanelViewerProps> = ({
  datasetState,
  maps,
}) => {
  const [windowWidth, setWindowWidth] = useState<number>(1.0);
  const [windowLevel, setWindowLevel] = useState<number>(0.0);
  const [zoom, setZoom] = useState<number>(1.0);
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(false);
  const [showTumorRoi, setShowTumorRoi] = useState<boolean>(true);
  const [showSarPanel, setShowSarPanel] = useState<boolean>(false);
  const [sigmaColormap, setSigmaColormap] = useState<ColormapName>("viridis");
  const [probe, setProbe] = useState<CursorProbe | null>(null);

  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});

  const width = datasetState?.width || 128;
  const height = datasetState?.height || 128;
  const fovXmm = (datasetState?.fov_x_m || 0.24) * 1000.0;
  const fovYmm = (datasetState?.fov_y_m || 0.24) * 1000.0;
  const b0Tesla = datasetState?.b0_tesla || 3.0;
  const sarMaxRange = b0Tesla >= 6.0 ? 10.5 : b0Tesla >= 2.5 ? 4.2 : 2.0;

  // Detectar coordenadas exactas y estadisticas de conductividad + SAR del ROI Tumoral en tiempo real
  const tumorRoiStats = useMemo(() => {
    const gt = maps.sigma_gt;
    const pred = maps.sigma;
    const sarPred = maps.sar;
    const sarGt = maps.sar_gt;
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    let sumPred = 0;
    let sumGt = 0;
    let sumSarPred = 0;
    let sumSarGt = 0;
    let minCol = width;
    let maxCol = 0;
    let minRow = height;
    let maxRow = 0;

    if (gt && gt.length === width * height) {
      const startCol = Math.floor(width * 0.54);
      for (let r = 0; r < height; r++) {
        for (let c = startCol; c < width; c++) {
          const idx = r * width + c;
          const gVal = gt[idx];
          if (gVal >= 0.92 && gVal <= 1.45) {
            sumX += c;
            sumY += r;
            count++;
            sumGt += gVal;
            if (pred && pred.length === width * height) {
              sumPred += pred[idx];
            }
            if (sarPred && sarPred.length === width * height) {
              sumSarPred += sarPred[idx];
            }
            if (sarGt && sarGt.length === width * height) {
              sumSarGt += sarGt[idx];
            }
            if (c < minCol) minCol = c;
            if (c > maxCol) maxCol = c;
            if (r < minRow) minRow = r;
            if (r > maxRow) maxRow = r;
          }
        }
      }
    }

    if (count > 5) {
      const cx = sumX / count;
      const cy = sumY / count;
      const rx = Math.max(6.5, (maxCol - minCol) * 0.5 + 1.8);
      const ry = Math.max(6.5, (maxRow - minRow) * 0.5 + 1.8);
      return {
        cx,
        cy,
        rx,
        ry,
        sigmaPredMean: sumPred / count,
        sigmaGtMean: sumGt / count,
        sarPredMean: sumSarPred / count,
        sarGtMean: sumSarGt / count,
      };
    }

    // Fallback analitico (cx = +0.24, cy = -0.18 en [-1, 1]^2)
    const fallbackCx = ((0.24 + 1.0) * 0.5) * (width - 1);
    const fallbackCy = ((-0.18 + 1.0) * 0.5) * (height - 1);
    return {
      cx: fallbackCx,
      cy: fallbackCy,
      rx: width * 0.08,
      ry: height * 0.075,
      sigmaPredMean: 0.0,
      sigmaGtMean: 1.2,
      sarPredMean: 0.0,
      sarGtMean: 2.95,
    };
  }, [maps.sigma_gt, maps.sigma, maps.sar, maps.sar_gt, width, height]);

  const panels: PanelSpec[] = [
    {
      id: "zf",
      title: "1. Zero-Filled Adjoint (Eᴴ s)",
      subtitle: `Sub-muestreo ${datasetState?.mask_type?.toUpperCase() || "CARTESIAN"} R=${
        datasetState?.acceleration || 4
      }x`,
      unit: "a.u.",
      matrix: maps.zero_filled_mag,
      minVal: 0.0,
      maxVal: 1.0,
      colormap: "grayscale",
      badgeText: `PSNR ${datasetState?.zero_filled_psnr?.toFixed(1) || "--"} dB`,
      badgeColor: "text-amber-300 border-amber-500/30 bg-amber-500/10",
    },
    {
      id: "anat",
      title: showGroundTruth
        ? "2. Anatomía Ground Truth |m_gt(x)|"
        : "2. PINN-SENSE Magnitud |m(x)|",
      subtitle: "Rama 1: SIREN Implícita (ω₀ = 30)",
      unit: "a.u.",
      matrix: showGroundTruth ? maps.m_mag_gt : maps.m_mag,
      minVal: 0.0,
      maxVal: 1.0,
      colormap: "grayscale",
      badgeText: showGroundTruth ? "REFERENCIA GT" : "DE-ALIASED SENSE",
      badgeColor: "text-cyan-300 border-cyan-500/30 bg-cyan-500/10",
    },
    {
      id: "sigma",
      title: showGroundTruth
        ? "3. Conductividad GT σ_gt(x)"
        : "3. Conductividad EPT σ(x)",
      subtitle: "Rama 3 + PDE Helmholtz (∇²φ⁺ / μ₀ω)",
      unit: "S/m",
      matrix: showGroundTruth ? maps.sigma_gt : maps.sigma,
      minVal: 0.01,
      maxVal: 1.75,
      colormap: sigmaColormap,
      badgeText: "Tumor σ ≈ 1.20 S/m",
      badgeColor: "text-emerald-300 border-emerald-500/30 bg-emerald-500/10",
    },
    showSarPanel
      ? {
          id: "eps",
          title: showGroundTruth
            ? "4. Dosimetría RF GT SAR_gt(x)"
            : "4. Dosimetría Térmica RF SAR(x)",
          subtitle: `Ley Ampère-Maxwell σ|E|²/2ρ @ ${b0Tesla.toFixed(1)}T`,
          unit: "W/kg",
          matrix: showGroundTruth ? maps.sar_gt : maps.sar,
          minVal: 0.0,
          maxVal: sarMaxRange,
          colormap: "inferno",
          badgeText: `Hotspot ${(
            (showGroundTruth
              ? tumorRoiStats.sarGtMean
              : tumorRoiStats.sarPredMean || tumorRoiStats.sarGtMean) || 0
          ).toFixed(2)} W/kg`,
          badgeColor: "text-orange-300 border-orange-500/40 bg-orange-500/15",
        }
      : {
          id: "eps",
          title: showGroundTruth
            ? "4. Permitividad Relativa GT εr_gt(x)"
            : "4. Permitividad Relativa εr(x)",
          subtitle: `Dispersión Dieléctrica @ ${
            datasetState?.larmor_freq_mhz?.toFixed(2) || "127.74"
          } MHz`,
          unit: "εr",
          matrix: showGroundTruth ? maps.eps_r_gt : maps.eps_r,
          minVal: 1.0,
          maxVal: 85.0,
          colormap: "magma",
          badgeText: "Rango [1 - 85]",
          badgeColor: "text-fuchsia-300 border-fuchsia-500/30 bg-fuchsia-500/10",
        },
  ];

  useEffect(() => {
    panels.forEach((p) => {
      const canvas = canvasRefs.current[p.id];
      if (!canvas || !p.matrix || p.matrix.length !== width * height) {
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const imgData = renderMatrixToImageData(
        p.matrix,
        width,
        height,
        p.minVal,
        p.maxVal,
        p.colormap,
        windowWidth,
        windowLevel
      );
      ctx.putImageData(imgData, 0, 0);

      // Dibujar contorno punteado cian/rojo del ROI Tumoral si esta activo
      if (showTumorRoi && tumorRoiStats) {
        ctx.save();
        // Halo rojo exterior sutil para alto contraste sobre viridis/magma/escala de grises
        ctx.beginPath();
        ctx.ellipse(
          tumorRoiStats.cx + 0.5,
          tumorRoiStats.cy + 0.5,
          tumorRoiStats.rx + 0.8,
          tumorRoiStats.ry + 0.8,
          0,
          0,
          Math.PI * 2
        );
        ctx.strokeStyle = "rgba(244, 63, 94, 0.92)";
        ctx.lineWidth = 1.25;
        ctx.setLineDash([3, 2]);
        ctx.stroke();

        // Anillo interior punteado cian
        ctx.beginPath();
        ctx.ellipse(
          tumorRoiStats.cx + 0.5,
          tumorRoiStats.cy + 0.5,
          tumorRoiStats.rx - 0.4,
          tumorRoiStats.ry - 0.4,
          0,
          0,
          Math.PI * 2
        );
        ctx.strokeStyle = "rgba(34, 211, 238, 0.95)";
        ctx.lineWidth = 1.0;
        ctx.setLineDash([2, 2]);
        ctx.lineDashOffset = 2;
        ctx.stroke();
        ctx.restore();
      }

      // Dibujar cursor sincronizado si esta activo
      if (probe) {
        ctx.save();
        ctx.strokeStyle = "rgba(34, 211, 238, 0.75)";
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(probe.col + 0.5, 0);
        ctx.lineTo(probe.col + 0.5, height);
        ctx.moveTo(0, probe.row + 0.5);
        ctx.lineTo(width, probe.row + 0.5);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(probe.col + 0.5, probe.row + 0.5, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = "#22d3ee";
        ctx.fill();
        ctx.restore();
      }
    });
  });

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;
    const col = Math.min(width - 1, Math.max(0, Math.floor(relX * width)));
    const row = Math.min(height - 1, Math.max(0, Math.floor(relY * height)));
    const idx = row * width + col;

    const x_mm = ((col / (width - 1)) * 2.0 - 1.0) * (fovXmm / 2.0);
    const y_mm = ((row / (height - 1)) * 2.0 - 1.0) * (fovYmm / 2.0);

    setProbe({
      row,
      col,
      x_mm,
      y_mm,
      zf_val: maps.zero_filled_mag ? maps.zero_filled_mag[idx] : 0,
      m_val: maps.m_mag ? maps.m_mag[idx] : 0,
      sigma_val: maps.sigma ? maps.sigma[idx] : 0,
      sigma_gt_val: maps.sigma_gt ? maps.sigma_gt[idx] : 0,
      eps_val: maps.eps_r ? maps.eps_r[idx] : 0,
      eps_gt_val: maps.eps_r_gt ? maps.eps_r_gt[idx] : 0,
      sar_val: maps.sar ? maps.sar[idx] : 0,
      sar_gt_val: maps.sar_gt ? maps.sar_gt[idx] : 0,
      phi_tr_val: maps.phi_tr ? maps.phi_tr[idx] : 0,
    });
  };

  const resetWindowLevel = () => {
    setWindowWidth(1.0);
    setWindowLevel(0.0);
    setZoom(1.0);
  };

  return (
    <div className="bg-slate-900/70 border border-slate-800/90 rounded-xl p-4 flex flex-col gap-3 shadow-2xl">
      {/* Barra de herramientas radiologica superior */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee]" />
          <h2 className="text-sm font-semibold tracking-wide text-slate-100 uppercase">
            Visor Sincronizado Multi-Modal PACS (SENSE + Helmholtz EPT)
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Selector de colormap para conductividad */}
          <div className="flex items-center gap-1.5 bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded-lg">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">LUT σ:</span>
            <select
              value={sigmaColormap}
              onChange={(e) => setSigmaColormap(e.target.value as ColormapName)}
              className="bg-transparent text-slate-200 font-medium focus:outline-none cursor-pointer"
            >
              <option value="viridis" className="bg-slate-900">
                Viridis
              </option>
              <option value="inferno" className="bg-slate-900">
                Inferno
              </option>
              <option value="turbo" className="bg-slate-900">
                Turbo
              </option>
            </select>
          </div>

          {/* Control Window / Level */}
          <div className="flex items-center gap-2 bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded-lg">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">W:</span>
            <input
              type="range"
              min="0.2"
              max="2.0"
              step="0.05"
              value={windowWidth}
              onChange={(e) => setWindowWidth(parseFloat(e.target.value))}
              className="w-16 accent-cyan-400 cursor-pointer"
            />
            <span className="text-slate-400">L:</span>
            <input
              type="range"
              min="-0.5"
              max="0.5"
              step="0.02"
              value={windowLevel}
              onChange={(e) => setWindowLevel(parseFloat(e.target.value))}
              className="w-16 accent-cyan-400 cursor-pointer"
            />
          </div>

          {/* Control Zoom */}
          <div className="flex items-center gap-1.5 bg-slate-950/90 border border-slate-800 px-2.5 py-1 rounded-lg">
            <Maximize2 className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">Zoom:</span>
            <input
              type="range"
              min="1.0"
              max="2.5"
              step="0.1"
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              className="w-14 accent-cyan-400 cursor-pointer"
            />
            <span className="font-mono text-slate-300">{zoom.toFixed(1)}x</span>
          </div>

          {/* Conmutador Panel 4: Permitividad vs Dosimetria SAR (W/kg) */}
          <button
            onClick={() => setShowSarPanel((v) => !v)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border font-medium transition cursor-pointer ${
              showSarPanel
                ? "bg-orange-500/20 border-orange-400/60 text-orange-200 shadow-[0_0_10px_rgba(249,115,22,0.25)]"
                : "bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700"
            }`}
          >
            <Flame className="w-3.5 h-3.5 text-orange-400" />
            <span>{showSarPanel ? "Viendo: SAR (W/kg)" : "Dosimetría SAR (W/kg)"}</span>
          </button>

          {/* Alternar Contorno ROI Tumor */}
          <button
            onClick={() => setShowTumorRoi((v) => !v)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border font-medium transition cursor-pointer ${
              showTumorRoi
                ? "bg-rose-500/20 border-rose-400/60 text-rose-200 shadow-[0_0_10px_rgba(244,63,94,0.2)]"
                : "bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700"
            }`}
          >
            <Target className="w-3.5 h-3.5 text-rose-400" />
            <span>Mostrar Contorno ROI Tumor</span>
          </button>

          {/* Alternar Prediccion vs Ground Truth */}
          <button
            onClick={() => setShowGroundTruth((v) => !v)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border font-medium transition ${
              showGroundTruth
                ? "bg-amber-500/20 border-amber-400/50 text-amber-200"
                : "bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700"
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            {showGroundTruth ? "Viendo: Ground Truth" : "Comparar con GT"}
          </button>

          <button
            onClick={resetWindowLevel}
            title="Restaurar Window/Level y Zoom"
            className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Grilla de 4 paneles sincronizados */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {panels.map((p) => (
          <div
            key={p.id}
            className="bg-slate-950 border border-slate-800/90 rounded-lg p-2.5 flex flex-col justify-between group hover:border-slate-700/80 transition"
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <div>
                <h3 className="text-xs font-semibold text-slate-100 leading-tight">
                  {p.title}
                </h3>
                <p className="text-[11px] text-slate-400">{p.subtitle}</p>
              </div>
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded border whitespace-nowrap ${p.badgeColor}`}
              >
                {p.badgeText}
              </span>
            </div>

            {/* Contenedor Canvas con Zoom y Cursor Sincronizado */}
            <div className="relative aspect-square w-full bg-black rounded border border-slate-900 overflow-hidden flex items-center justify-center">
              <canvas
                ref={(el) => {
                  canvasRefs.current[p.id] = el;
                }}
                width={width}
                height={height}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setProbe(null)}
                style={{
                  transform: `scale(${zoom})`,
                  imageRendering: "pixelated",
                }}
                className="w-full h-full object-contain cursor-crosshair transition-transform duration-75"
              />
              {/* Etiquetas de orientacion radiologica */}
              <span className="absolute top-1 left-1.5 text-[9px] font-mono text-slate-500 pointer-events-none">
                A
              </span>
              <span className="absolute bottom-1 left-1.5 text-[9px] font-mono text-slate-500 pointer-events-none">
                P
              </span>
              <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[9px] font-mono text-slate-500 pointer-events-none">
                R
              </span>
              <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[9px] font-mono text-slate-500 pointer-events-none">
                L
              </span>

              {/* Etiqueta flotante del ROI Tumoral con valor medio predicho vs Ground Truth */}
              {showTumorRoi && tumorRoiStats && (
                <div
                  style={{
                    left: `${Math.min(82, Math.max(18, (tumorRoiStats.cx / width) * 100))}%`,
                    top: `${Math.max(6, ((tumorRoiStats.cy - tumorRoiStats.ry - 4) / height) * 100)}%`,
                  }}
                  className="absolute -translate-x-1/2 -translate-y-full pointer-events-none z-10 bg-slate-950/90 backdrop-blur-sm border border-rose-500/60 rounded px-1.5 py-0.5 shadow-lg flex items-center gap-1 whitespace-nowrap"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-pulse" />
                  <span className="text-[9px] font-mono text-slate-100">
                    ROI σ̂:{" "}
                    <strong className="text-cyan-300">
                      {tumorRoiStats.sigmaPredMean > 0.01
                        ? tumorRoiStats.sigmaPredMean.toFixed(2)
                        : "--"}
                    </strong>{" "}
                    <span className="text-rose-300">
                      (GT: {tumorRoiStats.sigmaGtMean.toFixed(2)} S/m)
                    </span>
                  </span>
                </div>
              )}
            </div>

            {/* Barra de color cientifica y escala cuantitativa */}
            <div className="mt-2.5 flex flex-col gap-1">
              <div
                className="h-2 w-full rounded-sm border border-slate-800"
                style={{ background: getCssGradientForColormap(p.colormap) }}
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>
                  {p.minVal.toFixed(2)} {p.unit}
                </span>
                <span>
                  {((p.minVal + p.maxVal) * 0.5).toFixed(2)} {p.unit}
                </span>
                <span>
                  {p.maxVal.toFixed(2)} {p.unit}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Barra de telemetria sub-pixel e inspeccion fisica en tiempo real */}
      <div className="bg-slate-950/90 border border-slate-800/90 rounded-lg px-3.5 py-2 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center gap-2 text-cyan-400">
          <Crosshair className="w-4 h-4" />
          <span className="font-semibold">SONDA FÍSICA SINCRONIZADA:</span>
        </div>

        {probe ? (
          <div className="flex flex-wrap items-center gap-4 text-slate-200">
            <span>
              Voxel:{" "}
              <strong className="text-cyan-300">
                ({probe.col}, {probe.row})
              </strong>
            </span>
            <span>
              Posición FOV:{" "}
              <strong className="text-cyan-300">
                x={probe.x_mm >= 0 ? `+${probe.x_mm.toFixed(1)}` : probe.x_mm.toFixed(1)} mm, y=
                {probe.y_mm >= 0 ? `+${probe.y_mm.toFixed(1)}` : probe.y_mm.toFixed(1)} mm
              </strong>
            </span>
            <span>
              |m_ZF|: <strong className="text-amber-300">{probe.zf_val.toFixed(3)}</strong>
            </span>
            <span>
              |m_PINN|: <strong className="text-slate-100">{probe.m_val.toFixed(3)}</strong>
            </span>
            <span>
              σ(x):{" "}
              <strong className="text-emerald-400">
                {probe.sigma_val.toFixed(3)} S/m
              </strong>{" "}
              <span className="text-slate-500">
                (GT: {probe.sigma_gt_val.toFixed(3)} S/m)
              </span>
            </span>
            <span>
              εr(x):{" "}
              <strong className="text-fuchsia-400">
                {probe.eps_val.toFixed(1)}
              </strong>{" "}
              <span className="text-slate-500">
                (GT: {probe.eps_gt_val.toFixed(1)})
              </span>
            </span>
            <span>
              SAR(x):{" "}
              <strong className="text-orange-400">
                {probe.sar_val.toFixed(2)} W/kg
              </strong>{" "}
              <span className="text-slate-500">
                (GT: {probe.sar_gt_val.toFixed(2)} W/kg)
              </span>
            </span>
            <span>
              φ_tr:{" "}
              <strong className="text-sky-300">
                {probe.phi_tr_val.toFixed(3)} rad
              </strong>
            </span>
          </div>
        ) : (
          <span className="text-slate-500">
            Pasa el cursor sobre cualquiera de los 4 paneles para inspeccionar coordenadas milimétricas (mm) y propiedades eléctricas en S/m...
          </span>
        )}
      </div>
    </div>
  );
};
