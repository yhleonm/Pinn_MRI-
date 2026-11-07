import React, { useRef } from "react";
import {
  Activity,
  Database,
  FileUp,
  Play,
  RefreshCw,
  Square,
  Zap,
} from "lucide-react";
import { DatasetState, MaskType, TrainingConfig } from "../types";

interface ControlSidebarProps {
  config: TrainingConfig;
  setConfig: React.Dispatch<React.SetStateAction<TrainingConfig>>;
  datasetState: DatasetState | null;
  isTraining: boolean;
  isLoadingData: boolean;
  onStartTraining: () => void;
  onStopTraining: () => void;
  onRegeneratePhantom: (override?: Partial<TrainingConfig>) => void;
  onUploadH5: (file: File) => void;
  onLoadSampleH5: () => void;
}

export const ControlSidebar: React.FC<ControlSidebarProps> = ({
  config,
  setConfig,
  datasetState,
  isTraining,
  isLoadingData,
  onStartTraining,
  onStopTraining,
  onRegeneratePhantom,
  onUploadH5,
  onLoadSampleH5,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const larmorMHz = (42.58 * config.b0_tesla).toFixed(2);

  const handleFieldChange = (b0: number) => {
    setConfig((prev) => ({ ...prev, b0_tesla: b0 }));
    if (!isTraining) {
      onRegeneratePhantom({ b0_tesla: b0 });
    }
  };

  const handleMaskTypeChange = (mask_type: MaskType) => {
    setConfig((prev) => ({ ...prev, mask_type }));
    if (!isTraining) {
      onRegeneratePhantom({ mask_type });
    }
  };

  const handleAccelerationChange = (acceleration: number) => {
    setConfig((prev) => ({ ...prev, acceleration }));
    if (!isTraining) {
      onRegeneratePhantom({ acceleration });
    }
  };

  return (
    <aside className="w-full lg:w-80 shrink-0 bg-slate-900/80 border border-slate-800/90 rounded-xl p-4 flex flex-col gap-4 shadow-2xl">
      {/* Encabezado de control fisico */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-100">
            Parámetros Físicos & PINN
          </h2>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-300">
          {larmorMHz} MHz
        </span>
      </div>

      {/* Boton Principal Start / Stop Training */}
      <div className="flex flex-col gap-2">
        {!isTraining ? (
          <button
            onClick={onStartTraining}
            disabled={isLoadingData}
            className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition disabled:opacity-50 cursor-pointer"
          >
            <Play className="w-4 h-4 fill-current" />
            Iniciar Inversión PINN (GPU)
          </button>
        ) : (
          <button
            onClick={onStopTraining}
            className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 text-white font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg shadow-rose-600/25 transition cursor-pointer"
          >
            <Square className="w-4 h-4 fill-current" />
            Detener Entrenamiento
          </button>
        )}
      </div>

      {/* Seccion 1: Fuente de Datos K-Space (.h5 / Phantom) */}
      <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 flex flex-col gap-2.5">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          <span>1. Espacio-K Multi-Bobina</span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => onRegeneratePhantom()}
            disabled={isTraining || isLoadingData}
            className="flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-[11px] font-medium text-slate-200 transition disabled:opacity-50"
          >
            <RefreshCw className="w-3 h-3 text-cyan-400" />
            Phantom 8Ch
          </button>

          <button
            onClick={onLoadSampleH5}
            disabled={isTraining || isLoadingData}
            className="flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-700 text-[11px] font-medium text-slate-200 transition disabled:opacity-50"
          >
            <Database className="w-3 h-3 text-emerald-400" />
            Muestra .h5
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".h5,.hdf5"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUploadH5(file);
          }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isTraining || isLoadingData}
          className="w-full flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded bg-slate-900/60 hover:bg-slate-800 border border-dashed border-slate-700 text-[11px] text-slate-300 transition disabled:opacity-50"
        >
          <FileUp className="w-3.5 h-3.5 text-amber-400" />
          Subir Corte .h5 (fastMRI / Calgary)
        </button>
      </div>

      {/* Seccion 2: Campo Magnetico B0 y Sub-muestreo */}
      <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 flex flex-col gap-3 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 font-medium">Campo Estático B₀:</span>
          <div className="flex gap-1">
            {[1.5, 3.0, 7.0].map((b0) => (
              <button
                key={b0}
                disabled={isTraining}
                onClick={() => handleFieldChange(b0)}
                className={`px-2 py-0.5 rounded font-mono text-[11px] border transition ${
                  config.b0_tesla === b0
                    ? "bg-cyan-500/20 border-cyan-400 text-cyan-200 font-bold"
                    : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                {b0}T
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-slate-400 font-medium">
            Trayectoria K-Space P(k):
          </span>
          <div className="grid grid-cols-3 gap-1">
            {(["cartesian", "radial", "poisson"] as MaskType[]).map((mtype) => (
              <button
                key={mtype}
                disabled={isTraining}
                onClick={() => handleMaskTypeChange(mtype)}
                className={`py-1 rounded font-mono text-[11px] uppercase border transition ${
                  config.mask_type === mtype
                    ? "bg-cyan-500/20 border-cyan-400 text-cyan-200 font-bold"
                    : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                {mtype}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-400 font-medium">Aceleración R:</span>
          <div className="flex gap-1.5">
            {[2, 4, 8].map((r) => (
              <button
                key={r}
                disabled={isTraining}
                onClick={() => handleAccelerationChange(r)}
                className={`px-2.5 py-0.5 rounded font-mono text-[11px] border transition ${
                  config.acceleration === r
                    ? "bg-amber-500/20 border-amber-400 text-amber-200 font-bold"
                    : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                {r}x
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Banda Autocalibración (ACS):</span>
            <span className="font-mono text-cyan-300">{config.acs_lines} líneas</span>
          </div>
          <input
            type="range"
            min="8"
            max="32"
            step="2"
            disabled={isTraining}
            value={config.acs_lines}
            onChange={(e) => {
              const acs_lines = parseInt(e.target.value, 10);
              setConfig((p) => ({ ...p, acs_lines }));
            }}
            onMouseUp={() => onRegeneratePhantom({ acs_lines: config.acs_lines })}
            className="w-full accent-cyan-400"
          />
        </div>
      </div>

      {/* Seccion 3: Hiperparametros SIREN y Pesos PDE */}
      <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 flex flex-col gap-2.5 text-xs">
        <div className="flex items-center gap-1.5 font-semibold text-slate-200">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span>2. Arquitectura SIREN & PDE</span>
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Épocas Totales:</span>
            <span className="font-mono text-slate-200">{config.epochs}</span>
          </div>
          <input
            type="range"
            min="20"
            max="150"
            step="10"
            disabled={isTraining}
            value={config.epochs}
            onChange={(e) =>
              setConfig((p) => ({ ...p, epochs: parseInt(e.target.value, 10) }))
            }
            className="w-full accent-cyan-400"
          />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Warm-up Físico (λ_ept = 0):</span>
            <span className="font-mono text-amber-300">
              {config.warmup_epochs} épocas
            </span>
          </div>
          <input
            type="range"
            min="5"
            max="30"
            step="5"
            disabled={isTraining}
            value={config.warmup_epochs}
            onChange={(e) =>
              setConfig((p) => ({
                ...p,
                warmup_epochs: parseInt(e.target.value, 10),
              }))
            }
            className="w-full accent-amber-400"
          />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Frecuencia SIREN ω₀ (Anatomía):</span>
            <span className="font-mono text-cyan-300">
              {config.omega_anatomy.toFixed(0)}
            </span>
          </div>
          <input
            type="range"
            min="15"
            max="45"
            step="5"
            disabled={isTraining}
            value={config.omega_anatomy}
            onChange={(e) =>
              setConfig((p) => ({
                ...p,
                omega_anatomy: parseFloat(e.target.value),
              }))
            }
            className="w-full accent-cyan-400"
          />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Peso PDE Helmholtz (λ_σ):</span>
            <span className="font-mono text-emerald-300">
              {config.lambda_ept_sigma.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.02"
            max="0.50"
            step="0.02"
            disabled={isTraining}
            value={config.lambda_ept_sigma}
            onChange={(e) =>
              setConfig((p) => ({
                ...p,
                lambda_ept_sigma: parseFloat(e.target.value),
              }))
            }
            className="w-full accent-emerald-400"
          />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between">
            <span className="text-slate-400">Peso Armónico Laplace (λ_B0):</span>
            <span className="font-mono text-fuchsia-300">
              {config.lambda_harmonic_b0.toFixed(3)}
            </span>
          </div>
          <input
            type="range"
            min="0.005"
            max="0.10"
            step="0.005"
            disabled={isTraining}
            value={config.lambda_harmonic_b0}
            onChange={(e) =>
              setConfig((p) => ({
                ...p,
                lambda_harmonic_b0: parseFloat(e.target.value),
              }))
            }
            className="w-full accent-fuchsia-400"
          />
        </div>
      </div>

      {/* Resumen de salvaguardas fisicas activas */}
      <div className="bg-slate-950/90 border border-slate-800/80 rounded-lg p-2.5 text-[11px] text-slate-400 flex flex-col gap-1 font-mono">
        <div className="text-slate-200 font-semibold mb-0.5">
          SALVAGUARDAS FÍSICO-NUMÉRICAS:
        </div>
        <div>• Escalado cadena: (2/Lₓ)² ∇²_norm</div>
        <div>• Calibre fase VBC + Sobolev H²(Ω)</div>
        <div>• Aire denominador: |B₁⁺| + 10⁻³</div>
        <div>• Máscara tejido: |B₁⁺| &gt; 0.1·max</div>
        <div>
          • FOV: {((datasetState?.fov_x_m || 0.24) * 100).toFixed(0)} ×{" "}
          {((datasetState?.fov_y_m || 0.24) * 100).toFixed(0)} cm
        </div>
      </div>
    </aside>
  );
};
