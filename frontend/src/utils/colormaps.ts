import { ColormapName } from "../types";

/**
 * Decodifica una cadena Base64 IEEE-754 float32 (little-endian) en un Float32Array nativo.
 */
export function decodeFloat32Base64(b64: string): Float32Array {
  const binaryString = window.atob(b64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

type RGBStop = [number, [number, number, number]];

const COLORMAP_STOPS: Record<ColormapName, RGBStop[]> = {
  grayscale: [
    [0.0, [0, 0, 0]],
    [1.0, [255, 255, 255]],
  ],
  viridis: [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ],
  inferno: [
    [0.0, [4, 2, 12]],
    [0.25, [87, 16, 110]],
    [0.5, [188, 55, 84]],
    [0.75, [249, 142, 9]],
    [1.0, [252, 255, 164]],
  ],
  magma: [
    [0.0, [4, 2, 15]],
    [0.25, [81, 18, 124]],
    [0.5, [183, 55, 121]],
    [0.75, [251, 136, 97]],
    [1.0, [252, 253, 191]],
  ],
  turbo: [
    [0.0, [48, 18, 59]],
    [0.25, [40, 188, 235]],
    [0.5, [164, 252, 60]],
    [0.75, [251, 126, 33]],
    [1.0, [122, 4, 3]],
  ],
};

const LUT_CACHE: Partial<Record<ColormapName, Uint8Array>> = {};

export function getColormapLUT(name: ColormapName): Uint8Array {
  if (LUT_CACHE[name]) {
    return LUT_CACHE[name]!;
  }
  const lut = new Uint8Array(256 * 3);
  const stops = COLORMAP_STOPS[name];
  for (let i = 0; i < 256; i++) {
    const t = i / 255.0;
    let s0 = stops[0];
    let s1 = stops[stops.length - 1];
    for (let j = 0; j < stops.length - 1; j++) {
      if (t >= stops[j][0] && t <= stops[j + 1][0]) {
        s0 = stops[j];
        s1 = stops[j + 1];
        break;
      }
    }
    const span = Math.max(1e-6, s1[0] - s0[0]);
    const alpha = Math.min(1.0, Math.max(0.0, (t - s0[0]) / span));
    lut[i * 3 + 0] = Math.round(s0[1][0] + alpha * (s1[1][0] - s0[1][0]));
    lut[i * 3 + 1] = Math.round(s0[1][1] + alpha * (s1[1][1] - s0[1][1]));
    lut[i * 3 + 2] = Math.round(s0[1][2] + alpha * (s1[1][2] - s0[1][2]));
  }
  LUT_CACHE[name] = lut;
  return lut;
}

/**
 * Convierte una matriz Float32Array (H * W) en un ImageData RGBA aplicando
 * Window / Level radiologico y el colormap cientifico seleccionado.
 */
export function renderMatrixToImageData(
  matrix: Float32Array,
  width: number,
  height: number,
  minVal: number,
  maxVal: number,
  colormap: ColormapName,
  windowWidthFactor: number = 1.0,
  windowLevelOffset: number = 0.0
): ImageData {
  const lut = getColormapLUT(colormap);
  const imgData = new ImageData(width, height);
  const pixels = imgData.data;

  const baseRange = Math.max(1e-6, maxVal - minVal);
  const baseCenter = 0.5 * (minVal + maxVal);
  const effWidth = Math.max(1e-5, baseRange * windowWidthFactor);
  const effCenter = baseCenter + windowLevelOffset * baseRange;
  const low = effCenter - 0.5 * effWidth;

  const total = width * height;
  for (let i = 0; i < total; i++) {
    const v = matrix[i];
    const norm = (v - low) / effWidth;
    const idx = Math.min(255, Math.max(0, Math.round(norm * 255)));
    const pOffset = i * 4;
    const cOffset = idx * 3;
    pixels[pOffset + 0] = lut[cOffset + 0];
    pixels[pOffset + 1] = lut[cOffset + 1];
    pixels[pOffset + 2] = lut[cOffset + 2];
    pixels[pOffset + 3] = 255;
  }
  return imgData;
}

export function getCssGradientForColormap(name: ColormapName): string {
  const stops = COLORMAP_STOPS[name];
  const parts = stops.map(
    ([pos, [r, g, b]]) => `rgb(${r}, ${g}, ${b}) ${Math.round(pos * 100)}%`
  );
  return `linear-gradient(to right, ${parts.join(", ")})`;
}
