import { describe, expect, it } from 'vitest';
import registry from '../../shared/vessel_registry.json';
import { displayValue, percent, percentRange, riskColor, riskStops, validateMapping } from './mapping';
import type { Registry } from './types';
const contract = registry as Registry;
const names = Object.values(contract.targets).flatMap(t => t.meshes ?? []);
describe('anatomy contract', () => {
  it('maps every required mesh to one vessel', () => {
    const result = validateMapping(contract, names);
    expect(result.error).toBeNull(); expect(result.lookup.size).toBe(12);
    expect(result.lookup.get('D1')).toBe('LAD'); expect(result.lookup.get('PDA')).toBe('RCA');
  });
  it('rejects missing and duplicate geometry', () => {
    expect(validateMapping(contract, names.slice(1)).error).toContain(names[0]);
    expect(validateMapping(contract, [...names, names[0]]).error).toContain(names[0]);
  });
  it('rejects ambiguous target ownership', () => {
    const invalid = structuredClone(contract); invalid.targets.LCX.meshes!.push('D1');
    expect(validateMapping(invalid, names).error).toContain('Duplicate mapping');
  });
});
describe('probability display', () => {
  it('uses whole percents and never shows certainty', () => {
    expect(percent(0)).toBe('<1%');
    expect(percent(0.012)).toBe('1%');
    expect(percent(0.5)).toBe('50%');
    expect(percent(0.99)).toBe('99%');
    expect(percent(0.996)).toBe('>99%');
    expect(percent(1)).toBe('>99%');
  });
  it('collapses an interval that rounds to one value', () => {
    expect(percentRange(0.62, 0.77)).toBe('62% to 77%');
    expect(percentRange(0.995, 0.999)).toBe('>99%');
  });
});
describe('risk ramp', () => {
  const luminance = (hex: string) => {
    const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map(c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  it('starts and ends on the published stops', () => {
    expect(riskColor(0)).toBe(riskStops[0]);
    expect(riskColor(1)).toBe(riskStops[riskStops.length - 1]);
    expect(riskColor(Number.NaN)).toBe(riskStops[0]);
  });
  it('gets brighter as risk rises', () => {
    const steps = Array.from({ length: 21 }, (_, i) => luminance(riskColor(i / 20)));
    steps.slice(1).forEach((value, i) => expect(value).toBeGreaterThan(steps[i]));
  });
});
describe('clinical values', () => {
  it('reads flags, options and numbers the way a clinician would', () => {
    expect(displayValue({ type: 'categorical', option_labels: { Fmale: 'Female' } }, 'Fmale')).toBe('Female');
    expect(displayValue({ type: 'binary' }, 1)).toBe('Yes');
    expect(displayValue({ type: 'binary' }, 0)).toBe('No');
    expect(displayValue({ type: 'yn' }, 'Y')).toBe('Yes');
    expect(displayValue({ type: 'number', unit: 'kg/m2' }, 29.0657)).toBe('29.1 kg/m2');
    expect(displayValue({ type: 'number', unit: 'mg/dL' }, 240, false)).toBe('240');
    expect(displayValue({ type: 'categorical', option_labels: { N: 'None' } }, 'N')).toBe('None');
  });
});
