import type { FeatureSpec, Registry, Vessel } from './types';
export const vessels: Vessel[] = ['LAD', 'LCX', 'RCA'];
export function validateMapping(registry: Registry, meshNames: string[]) {
  const lookup = new Map<string, Vessel>();
  const errors: string[] = [];
  for (const vessel of vessels) {
    const names = registry.targets[vessel]?.meshes;
    if (!names?.length) errors.push(`No mesh mapping for ${vessel}`);
    for (const name of names ?? []) {
      if (lookup.has(name)) errors.push(`Duplicate mapping: ${name}`);
      if (meshNames.filter(n => n === name).length !== 1) errors.push(`Expected exactly one mesh: ${name}`);
      lookup.set(name, vessel);
    }
  }
  return { lookup, error: errors.length ? errors.join('. ') : null };
}

/** Whole percents. The ends read "<1%" and ">99%" so a model output never looks certain. */
export function percent(value: number) {
  if (value < 0.01) return '<1%';
  if (value > 0.99) return '>99%';
  return `${Math.round(value * 100)}%`;
}

export function percentRange(low: number, high: number) {
  const a = percent(low), b = percent(high);
  return a === b ? a : `${a} to ${b}`;
}

/**
 * Risk ramp shared by the 3D vessels, the vessel bars and the legend: blue, warm grey, yellow.
 * Luminance rises with risk, the blue–yellow axis survives common color-vision deficiencies, and
 * neither end is red or orange, which would disappear against the myocardium.
 */
export const riskStops = ['#3f6fb5', '#a6a49a', '#ffd43b'] as const;
export const riskGradient = `linear-gradient(90deg, ${riskStops.join(', ')})`;

const channels = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));

export function riskColor(probability: number) {
  const p = Number.isFinite(probability) ? Math.min(1, Math.max(0, probability)) : 0;
  const t = p * (riskStops.length - 1);
  const i = Math.min(Math.floor(t), riskStops.length - 2);
  const from = channels(riskStops[i]), to = channels(riskStops[i + 1]);
  return '#' + from.map((v, k) => Math.round(v + (to[k] - v) * (t - i)).toString(16).padStart(2, '0')).join('');
}

type ValueSpec = Pick<FeatureSpec, 'type' | 'unit' | 'option_labels'>;

/** Clinical reading of a stored value: Yes/No for flags, option labels, one decimal, unit. */
export function displayValue(spec: ValueSpec | undefined, value: unknown, withUnit = true): string {
  if (value == null || value === '') return '—';
  const label = spec?.option_labels?.[String(value)];
  if (label) return label;
  if (spec?.type === 'binary') return Number(value) === 1 ? 'Yes' : 'No';
  if (spec?.type === 'yn') return value === 'Y' ? 'Yes' : 'No';
  if (typeof value === 'number') {
    const text = Number.isInteger(value) ? String(value) : value.toFixed(1);
    return withUnit && spec?.unit ? `${text} ${spec.unit}` : text;
  }
  return String(value);
}
