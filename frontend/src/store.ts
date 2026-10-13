import { create } from 'zustand';
import type { Analysis, Registry, Sample, Schema, Vessel } from './types';
const base = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}/api${path}`, init);
  if (!res.ok) throw new Error(`API request failed (${res.status}). Please retry.`);
  return res.json() as Promise<T>;
}
let active: AbortController | undefined;
let sequence = 0;
interface State {
  registry: Registry | null; schema: Schema | null; samples: Sample[];
  analysis: Analysis | null; sampleId: string; selected: Vessel | null;
  mode: 'anatomy' | 'risk'; loading: boolean; initializing: boolean;
  error: string | null; initError: string | null;
  initialize: () => Promise<void>; selectSample: (id: string) => Promise<void>;
  selectVessel: (v: Vessel | null) => void; setMode: (mode: 'anatomy' | 'risk') => void;
}
export const useStore = create<State>((set, get) => ({
  registry: null, schema: null, samples: [], analysis: null, sampleId: '', selected: null,
  mode: 'anatomy', loading: false, initializing: true, error: null, initError: null,
  initialize: async () => {
    set({ initializing: true, initError: null });
    try {
      const [registry, schema, samples] = await Promise.all([
        request<Registry>('/targets'), request<Schema>('/schema'), request<Sample[]>('/samples'),
      ]);
      set({ registry, schema, samples, initializing: false });
    } catch (e) { set({ initializing: false, initError: (e as Error).message }); }
  },
  selectSample: async (id) => {
    active?.abort();
    const version = ++sequence;
    active = new AbortController();
    set({ sampleId: id, analysis: null, error: null, loading: !!id, mode: 'anatomy' });
    if (!id) return;
    const sample = get().samples.find(s => s.id === id);
    if (!sample) { set({ loading: false, error: 'Sample patient is unavailable.' }); return; }
    try {
      const analysis = await request<Analysis>('/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sample.features), signal: active.signal,
      });
      if (version === sequence) set({ analysis, loading: false, mode: 'risk' });
    } catch (e) {
      if (version === sequence && (e as Error).name !== 'AbortError') set({ loading: false, error: (e as Error).message });
    }
  },
  selectVessel: selected => set({ selected }),
  setMode: mode => set({ mode }),
}));
