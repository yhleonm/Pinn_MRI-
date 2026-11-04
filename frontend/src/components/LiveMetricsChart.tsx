import React from "react";
import { Activity, Award, Flame, TrendingDown } from "lucide-react";
import { DatasetState, EpochTelemetry } from "../types";

interface LiveMetricsChartProps {
  history: EpochTelemetry[];
  datasetState: DatasetState | null;
  warmupEpochs: number;
}

function buildPolyline(
  values: number[],
  width: number,
  height: number,
  minVal: number,
  maxVal: number
): string {
  if (values.length === 0) return "";
  if (values.length === 1) {
    const y =
      height -
      ((values[0] - minVal) / Math.max(1e-8, maxVal - minVal)) * (height - 16) -
      8;
    return `0,${y.toFixed(1)} ${width},${y.toFixed(1)}`;
  }
  const span = Math.max(1e-8, maxVal - minVal);
  return values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const norm = Math.min(1.0, Math.max(0.0, (v - minVal) / span));
      const y = height - 8 - norm * (height - 16);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export const LiveMetricsChart: React.FC<LiveMetricsChartProps> = ({
  history,
  datasetState,
  warmupEpochs,
}) => {
  const latest = history.length > 0 ? history[history.length - 1] : null;
  const zfPsnr = datasetState?.zero_filled_psnr || 27.6;

  // Series en escala log10 para perdidas fisicas
  const logKspace = history.map((h) => Math.log10(Math.max(1e-5, h.loss_kspace)));
  const logHelm = history.map((h) => Math.log10(Math.max(1e-5, h.loss_helmholtz)));
  const logHarm = history.map((h) => Math.log10(Math.max(1e-5, h.loss_harmonic)));
  const allLogs = [...logKspace, ...logHelm, ...logHarm, -3.0, 0.8];
  const minLog = Math.min(...allLogs);
  const maxLog = Math.max(...allLogs);

  // Series para PSNR y NRMSE_sigma
  const psnrSeries = history.map((h) => h.psnr_db);
  const nrmseSeries = history.map((h) => h.nrmse_sigma);

  const svgW = 460;
  const svgH = 140;

  const totalEp = latest?.total_epochs || Math.max(50, history.length);
  const warmupX = Math.min(
    svgW,
    Math.max(0, (warmupEpochs / Math.max(1, totalEp)) * svgW)
  );

  return (
    <div className="bg-slate-900/70 border border-slate-800/90 rounded-xl p-4 flex flex-col gap-3 shadow-xl">
      {/* Tarjetas de KPIs en vivo */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block">
              PSNR Reconstrucción |m(x)|
            </span>
            <div className="text-lg font-bold font-mono text-cyan-300">
              {latest ? `${latest.psnr_db.toFixed(2)} dB` : `${zfPsnr.toFixed(2)} dB`}
            </div>
            <span className="text-[10px] font-mono text-emerald-400">
              {latest
                ? `${latest.psnr_db >= zfPsnr ? "+" : ""}${(
                    latest.psnr_db - zfPsnr
                  ).toFixed(2)} dB vs Zero-Filled`
                : "Base Zero-Filled"}
            </span>
          </div>
          <Award className="w-7 h-7 text-cyan-400/40" />
        </div>

        <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block">
              SSIM Estructural
            </span>
            <div className="text-lg font-bold font-mono text-teal-300">
              {latest
                ? latest.ssim.toFixed(4)
                : (datasetState?.zero_filled_ssim || 0.69).toFixed(4)}
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              Base ZF: {(datasetState?.zero_filled_ssim || 0.69).toFixed(4)}
            </span>
          </div>
          <Activity className="w-7 h-7 text-teal-400/40" />
        </div>

        <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block">
              NRMSE Conductividad σ(x)
            </span>
            <div className="text-lg font-bold font-mono text-emerald-400">
              {latest ? latest.nrmse_sigma.toFixed(4) : "--"}
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              Tumor σ:{" "}
              <strong className="text-emerald-300">
                {latest ? `${latest.tumor_sigma_mean.toFixed(2)} S/m` : "1.20 S/m (GT)"}
              </strong>
            </span>
          </div>
          <Flame className="w-7 h-7 text-emerald-400/40" />
        </div>

        <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-3 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block">
              Residuo PDE Helmholtz
            </span>
            <div className="text-lg font-bold font-mono text-amber-300">
              {latest ? latest.loss_helmholtz.toFixed(4) : "--"}
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              Estado:{" "}
              <strong
                className={
                  latest?.ept_active ? "text-emerald-400" : "text-amber-300"
                }
              >
                {latest
                  ? latest.ept_active
                    ? "PDE ACOPLADA"
                    : "WARM-UP (λ=0)"
                  : "EN ESPERA"}
              </strong>
            </span>
          </div>
          <TrendingDown className="w-7 h-7 text-amber-400/40" />
        </div>
      </div>

      {/* Graficos SVG dobles de convergencia */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Grafico 1: Perdidas Fisicas (Log10) */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-200">
              Convergencia de Pérdidas Físicas (log₁₀)
            </span>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="text-cyan-400">● L_kspace</span>
              <span className="text-emerald-400">● L_helmholtz</span>
              <span className="text-fuchsia-400">● L_harmonic</span>
            </div>
          </div>

          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="w-full h-36 bg-slate-950/60 rounded border border-slate-900 overflow-visible"
          >
            {/* Linea divisoria de Warm-up -> EPT-ON */}
            <line
              x1={warmupX}
              y1={0}
              x2={warmupX}
              y2={svgH}
              stroke="rgba(251, 191, 36, 0.35)"
              strokeDasharray="4 4"
              strokeWidth="1.2"
            />
            <text
              x={warmupX + 4}
              y={14}
              fill="rgba(251, 191, 36, 0.75)"
              fontSize="9"
              fontFamily="monospace"
            >
              Warm-up → EPT ON (Ep {warmupEpochs})
            </text>

            {history.length > 0 && (
              <>
                <polyline
                  fill="none"
                  stroke="#e879f9"
                  strokeWidth="1.6"
                  points={buildPolyline(logHarm, svgW, svgH, minLog, maxLog)}
                />
                <polyline
                  fill="none"
                  stroke="#34d399"
                  strokeWidth="2.0"
                  points={buildPolyline(logHelm, svgW, svgH, minLog, maxLog)}
                />
                <polyline
                  fill="none"
                  stroke="#22d3ee"
                  strokeWidth="2.2"
                  points={buildPolyline(logKspace, svgW, svgH, minLog, maxLog)}
                />
              </>
            )}
          </svg>
        </div>

        {/* Grafico 2: Metricas Cuantitativas (PSNR dB y NRMSE_sigma) */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-200">
              Calidad Anatómica (PSNR) vs Error Conductividad (NRMSE σ)
            </span>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="text-cyan-400">● PSNR [5 - 34 dB]</span>
              <span className="text-amber-400">● NRMSE σ [0.2 - 1.0]</span>
            </div>
          </div>

          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="w-full h-36 bg-slate-950/60 rounded border border-slate-900 overflow-visible"
          >
            <line
              x1={warmupX}
              y1={0}
              x2={warmupX}
              y2={svgH}
              stroke="rgba(251, 191, 36, 0.35)"
              strokeDasharray="4 4"
              strokeWidth="1.2"
            />
            {history.length > 0 && (
              <>
                <polyline
                  fill="none"
                  stroke="#22d3ee"
                  strokeWidth="2.2"
                  points={buildPolyline(psnrSeries, svgW, svgH, 5.0, 34.0)}
                />
                <polyline
                  fill="none"
                  stroke="#fbbf24"
                  strokeWidth="2.0"
                  points={buildPolyline(nrmseSeries, svgW, svgH, 0.2, 0.9)}
                />
              </>
            )}
          </svg>
        </div>
      </div>
    </div>
  );
};
