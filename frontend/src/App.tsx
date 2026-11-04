import React from "react";
import { Activity, Cpu, ShieldCheck, Wifi, WifiOff } from "lucide-react";
import { ControlSidebar } from "./components/ControlSidebar";
import { KSpaceInspector } from "./components/KSpaceInspector";
import { LiveMetricsChart } from "./components/LiveMetricsChart";
import { MultiPanelViewer } from "./components/MultiPanelViewer";
import { usePINNWebSocket } from "./hooks/usePINNWebSocket";

export const App: React.FC = () => {
  const {
    datasetState,
    maps,
    config,
    setConfig,
    history,
    isTraining,
    wsConnected,
    statusMessage,
    isLoadingData,
    startTraining,
    stopTraining,
    regeneratePhantom,
    uploadH5File,
    loadSampleH5,
  } = usePINNWebSocket();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Barra superior estilo Estacion Radiologica PACS */}
      <header className="border-b border-slate-800/90 bg-slate-900/90 backdrop-blur px-6 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/15 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-slate-100">
                MEDICAL INSIGHT MRI
              </h1>
              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/30 text-cyan-300">
                Decoupled PINN-SENSE + Helmholtz EPT
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Reconstrucción Acelerada desde Espacio-K Multi-Bobina y Mapeo Cuantitativo de Propiedades Eléctricas (σ, εr)
            </p>
          </div>
        </div>

        {/* Estado de GPU y WebSocket */}
        <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-950 border border-slate-800 text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>{datasetState?.gpu_name || "NVIDIA GeForce RTX 4060 Ti"}</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-950 border border-slate-800 text-slate-300">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>Autograd 2° Orden FP32</span>
          </div>

          <div
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full border ${
              wsConnected
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                : "bg-rose-500/10 border-rose-500/30 text-rose-300"
            }`}
          >
            {wsConnected ? (
              <Wifi className="w-3.5 h-3.5" />
            ) : (
              <WifiOff className="w-3.5 h-3.5" />
            )}
            <span>{wsConnected ? "WS /ws/train ONLINE" : "WS OFFLINE"}</span>
          </div>
        </div>
      </header>

      {/* Cinta de estado operativo en tiempo real */}
      <div className="bg-slate-900/50 border-b border-slate-800/70 px-6 py-1.5 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-2 text-slate-300 truncate">
          <span
            className={`h-2 w-2 rounded-full ${
              isTraining
                ? "bg-emerald-400 animate-ping"
                : "bg-cyan-400"
            }`}
          />
          <span>{statusMessage}</span>
        </div>
        {history.length > 0 && (
          <span className="text-slate-400 shrink-0">
            Época {history[history.length - 1].epoch} /{" "}
            {history[history.length - 1].total_epochs}
          </span>
        )}
      </div>

      {/* Cuerpo principal de la estacion de trabajo */}
      <main className="flex-1 p-5 flex flex-col lg:flex-row gap-5 max-w-[1920px] w-full mx-auto">
        <ControlSidebar
          config={config}
          setConfig={setConfig}
          datasetState={datasetState}
          isTraining={isTraining}
          isLoadingData={isLoadingData}
          onStartTraining={startTraining}
          onStopTraining={stopTraining}
          onRegeneratePhantom={regeneratePhantom}
          onUploadH5={uploadH5File}
          onLoadSampleH5={loadSampleH5}
        />

        <div className="flex-1 flex flex-col gap-5 min-w-0">
          <MultiPanelViewer datasetState={datasetState} maps={maps} />
          <KSpaceInspector datasetState={datasetState} maps={maps} />
          <LiveMetricsChart
            history={history}
            datasetState={datasetState}
            warmupEpochs={config.warmup_epochs}
          />
        </div>
      </main>
    </div>
  );
};

export default App;
