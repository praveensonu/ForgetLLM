export function parseRmuLayers(value: string): number[] | null {
  const parts = value.split(",").map((part) => part.trim());
  if (!parts.length || parts.some((part) => !/^\d+$/.test(part))) return null;
  const layers = parts.map(Number);
  if (layers.some((layer) => !Number.isSafeInteger(layer)) || new Set(layers).size !== layers.length) return null;
  return layers.sort((a, b) => a - b);
}

export function rmuTrainableParamsRegex(layers: number[]): string {
  return String.raw`.*layers\.(${layers.join("|")})\..*`;
}

export function defaultRmuModuleRegex(layers: number[]): string {
  return String.raw`.*layers\.${Math.max(...layers)}$`;
}
