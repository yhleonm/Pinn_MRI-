export type MaskType = "cartesian" | "radial" | "poisson";

export type ColormapName = "grayscale" | "viridis" | "inferno" | "magma" | "turbo";

export interface DatasetState {
  source: string;
  height: number;
  width: number;
  num_coils: number;
  b0_tesla: number;
  larmor_freq_mhz: number;
  fov_x_m: number;
  fov_y_m: number;
  mask_type: MaskType;
  acceleration: number;
  acs_lines: number;
  sampling_fraction: number;
  zero_filled_psnr: number;
  zero_filled_ssim: number;
  gpu_name: string;
  vram_allocated_mb: number;
  maps_f32_b64: Record<string, string>;
}

export interface DecodedMaps {
  zero_filled_mag?: Float32Array;
  m_mag?: Float32Array;
  sigma?: Float32Array;
  eps_r?: Float32Array;
  sar?: Float32Array;
  phi_tr?: Float32Array;
  phi_b0?: Float32Array;
  m_mag_gt?: Float32Array;
  sigma_gt?: Float32Array;
  eps_r_gt?: Float32Array;
  sar_gt?: Float32Array;
  phi_tr_gt?: Float32Array;
  phi_b0_gt?: Float32Array;
  kspace_log_full?: Float32Array;
  kspace_log_masked?: Float32Array;
  mask?: Float32Array;
  tissue_mask?: Float32Array;
}

export interface TrainingConfig {
  b0_tesla: number;
  mask_type: MaskType;
  acceleration: number;
  acs_lines: number;
  epochs: number;
  warmup_epochs: number;
  omega_anatomy: number;
  omega_coil: number;
  omega_ept: number;
  lr_anatomy: number;
  lr_coil: number;
  lr_ept: number;
  lambda_kspace: number;
  lambda_ept_sigma: number;
  lambda_ept_eps: number;
  lambda_coil_smooth: number;
  lambda_harmonic_b0: number;
  telemetry_interval: number;
}

export interface EpochTelemetry {
  type: "epoch" | "completed" | "stopped" | "error";
  epoch: number;
  total_epochs: number;
  total_loss: number;
  loss_kspace: number;
  loss_helmholtz: number;
  loss_coil: number;
  loss_harmonic: number;
  psnr_db: number;
  ssim: number;
  nrmse_sigma: number;
  tumor_sigma_mean: number;
  ept_active: boolean;
  elapsed_ms: number;
  maps_f32_b64?: Record<string, string> | null;
  message?: string | null;
}

export interface CursorProbe {
  row: number;
  col: number;
  x_mm: number;
  y_mm: number;
  zf_val: number;
  m_val: number;
  sigma_val: number;
  sigma_gt_val: number;
  eps_val: number;
  eps_gt_val: number;
  sar_val: number;
  sar_gt_val: number;
  phi_tr_val: number;
}
