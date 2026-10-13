import type { Registry, Vessel } from './types';
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
export const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
