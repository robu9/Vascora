import { describe, expect, it } from 'vitest';
import registry from '../../shared/vessel_registry.json';
import { percent, validateMapping } from './mapping';
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
  it('preserves a zero probability', () => expect(percent(0)).toBe('0.0%'));
});
