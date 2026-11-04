import { useCallback, useEffect, useRef, useState } from "react";
import {
  DatasetState,
  DecodedMaps,
  EpochTelemetry,
  TrainingConfig,
} from "../types";
import { decodeFloat32Base64 } from "../utils/colormaps";

const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/train";

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  b0_tesla: 3.0,
  mask_type: "cartesian",
  acceleration: 4,
  acs_lines: 18,
  epochs: 60,
  warmup_epochs: 15,
  omega_anatomy: 30.0,
  omega_coil: 5.0,
  omega_ept: 15.0,
  lr_anatomy: 7.0e-4,
  lr_coil: 1.2e-3,
  lr_ept: 2.5e-3,
  lambda_kspace: 1.0,
  lambda_ept_sigma: 0.15,
  lambda_ept_eps: 0.02,
  lambda_coil_smooth: 0.02,
  lambda_harmonic_b0: 0.02,
  telemetry_interval: 2,
};

export function usePINNWebSocket() {
  const [datasetState, setDatasetState] = useState<DatasetState | null>(null);
  const [maps, setMaps] = useState<DecodedMaps>({});
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [history, setHistory] = useState<EpochTelemetry[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>(
    "Conectando con el servidor FastAPI en GPU..."
  );
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);

  const applyEncodedMaps = useCallback((encoded: Record<string, string>) => {
    setMaps((prev) => {
      const next: DecodedMaps = { ...prev };
      for (const [key, b64] of Object.entries(encoded)) {
        if (b64) {
          (next as Record<string, Float32Array>)[key] = decodeFloat32Base64(b64);
        }
      }
      return next;
    });
  }, []);

  const fetchInitialState = useCallback(async () => {
    try {
      setIsLoadingData(true);
      const res = await fetch(`${API_BASE}/api/state`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data: DatasetState = await res.json();
      setDatasetState(data);
      applyEncodedMaps(data.maps_f32_b64);
      setConfig((prev) => ({
        ...prev,
        b0_tesla: data.b0_tesla,
        mask_type: data.mask_type,
        acceleration: data.acceleration,
        acs_lines: data.acs_lines,
      }));
      setStatusMessage(
        `Espacio-k listo en ${data.gpu_name} (${data.height}x${data.width}, ${data.num_coils} bobinas, R=${data.acceleration}x)`
      );
    } catch (err) {
      setStatusMessage(
        `Error conectando con backend (${API_BASE}): ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setIsLoadingData(false);
    }
  }, [applyEncodedMaps]);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const payload: EpochTelemetry = JSON.parse(event.data);
        if (payload.type === "error") {
          setIsTraining(false);
          setStatusMessage(`Error en GPU: ${payload.message || "Fallo desconocido"}`);
          return;
        }
        if (payload.maps_f32_b64) {
          applyEncodedMaps(payload.maps_f32_b64);
        }
        if (payload.type === "epoch" || payload.type === "completed") {
          setHistory((prev) => [...prev, payload]);
        }
        if (payload.type === "completed") {
          setIsTraining(false);
          setStatusMessage(
            `Entrenamiento completado (${payload.epoch} epocas) | PSNR: ${payload.psnr_db.toFixed(
              2
            )} dB | SSIM: ${payload.ssim.toFixed(4)} | NRMSE σ: ${payload.nrmse_sigma.toFixed(4)}`
          );
        } else if (payload.type === "stopped") {
          setIsTraining(false);
          setStatusMessage(
            payload.message || `Entrenamiento detenido en epoca ${payload.epoch}.`
          );
        } else {
          const modeLabel = payload.ept_active
            ? "HELMHOLTZ EPT ACOPLADO"
            : "WARM-UP ANATOMICO (λ_ept = 0)";
          setStatusMessage(
            `[Epoca ${payload.epoch}/${payload.total_epochs}] ${modeLabel} | PSNR: ${payload.psnr_db.toFixed(
              2
            )} dB | NRMSE σ: ${payload.nrmse_sigma.toFixed(4)} (${payload.elapsed_ms.toFixed(0)} ms)`
          );
        }
      } catch (err) {
        console.error("Error decodificando telemetria WS:", err);
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
      setIsTraining(false);
    };
  }, [applyEncodedMaps]);

  useEffect(() => {
    fetchInitialState();
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchInitialState, connectWebSocket]);

  const startTraining = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWebSocket();
      setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          setHistory([]);
          setIsTraining(true);
          wsRef.current.send(JSON.stringify({ action: "start", config }));
        }
      }, 250);
      return;
    }
    setHistory([]);
    setIsTraining(true);
    wsRef.current.send(JSON.stringify({ action: "start", config }));
  }, [config, connectWebSocket]);

  const stopTraining = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "stop" }));
    }
  }, []);

  const regeneratePhantom = useCallback(
    async (overrideCfg?: Partial<TrainingConfig>) => {
      const nextCfg = { ...config, ...(overrideCfg || {}) };
      setConfig(nextCfg);
      try {
        setIsLoadingData(true);
        setStatusMessage("Generando Phantom Shepp-Logan 8-bobinas y aplicando mascara en GPU...");
        const res = await fetch(`${API_BASE}/api/phantom`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            height: 128,
            width: 128,
            num_coils: 8,
            b0_tesla: nextCfg.b0_tesla,
            fov_x_m: 0.24,
            fov_y_m: 0.24,
            noise_snr_db: 35.0,
            mask_type: nextCfg.mask_type,
            acceleration: nextCfg.acceleration,
            acs_lines: nextCfg.acs_lines,
          }),
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data: DatasetState = await res.json();
        setDatasetState(data);
        applyEncodedMaps(data.maps_f32_b64);
        setHistory([]);
        setStatusMessage(
          `Phantom regenerado: ${data.mask_type.toUpperCase()} R=${data.acceleration}x | B0=${data.b0_tesla}T (${data.larmor_freq_mhz} MHz)`
        );
      } catch (err) {
        setStatusMessage(`Error al regenerar phantom: ${String(err)}`);
      } finally {
        setIsLoadingData(false);
      }
    },
    [config, applyEncodedMaps]
  );

  const uploadH5File = useCallback(
    async (file: File) => {
      try {
        setIsLoadingData(true);
        setStatusMessage(`Cargando volumen ${file.name} en GPU...`);
        const formData = new FormData();
        formData.append("file", file);
        const params = new URLSearchParams({
          mask_type: config.mask_type,
          acceleration: String(config.acceleration),
          acs_lines: String(config.acs_lines),
          b0_tesla: String(config.b0_tesla),
        });
        const res = await fetch(`${API_BASE}/api/upload-h5?${params.toString()}`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          throw new Error(errJson.detail || `HTTP ${res.status}`);
        }
        const data: DatasetState = await res.json();
        setDatasetState(data);
        applyEncodedMaps(data.maps_f32_b64);
        setHistory([]);
        setStatusMessage(
          `Archivo .h5 cargado: ${data.source} (${data.height}x${data.width}, ${data.num_coils} bobinas)`
        );
      } catch (err) {
        setStatusMessage(`Error al subir .h5: ${String(err)}`);
      } finally {
        setIsLoadingData(false);
      }
    },
    [config, applyEncodedMaps]
  );

  const loadSampleH5 = useCallback(async () => {
    try {
      setIsLoadingData(true);
      setStatusMessage("Cargando muestra sample_brain_8coils.h5 desde disco...");
      const res = await fetch(`${API_BASE}/api/load-sample-h5`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          height: 128,
          width: 128,
          num_coils: 8,
          b0_tesla: config.b0_tesla,
          fov_x_m: 0.24,
          fov_y_m: 0.24,
          noise_snr_db: 35.0,
          mask_type: config.mask_type,
          acceleration: config.acceleration,
          acs_lines: config.acs_lines,
        }),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data: DatasetState = await res.json();
      setDatasetState(data);
      applyEncodedMaps(data.maps_f32_b64);
      setHistory([]);
      setStatusMessage(`Archivo .h5 cargado: ${data.source} (R=${data.acceleration}x)`);
    } catch (err) {
      setStatusMessage(`Error cargando .h5 de muestra: ${String(err)}`);
    } finally {
      setIsLoadingData(false);
    }
  }, [config, applyEncodedMaps]);

  return {
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
  };
}
