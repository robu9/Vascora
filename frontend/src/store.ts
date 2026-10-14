import { create } from 'zustand';
import type { Analysis, Counterfactual, Features, MetricsReport, Registry, Sample, Schema, Target, Vessel } from './types';

const base = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const targets: Target[] = ['CAD', 'LAD', 'LCX', 'RCA'];

interface ValidationIssue { loc?: (string | number)[]; msg?: string }

/** FastAPI puts a message or a list of field issues in `detail`; surface it when present. */
async function failure(res: Response): Promise<Error> {
  let detail: unknown;
  try { detail = ((await res.json()) as { detail?: unknown }).detail; } catch { detail = undefined; }
  if (typeof detail === 'string' && detail) return new Error(detail);
  if (Array.isArray(detail) && detail.length) {
    return new Error((detail as ValidationIssue[]).map(issue => `${issue.loc?.at(-1) ?? 'input'}: ${issue.msg ?? 'invalid value'}`).join('; '));
  }
  return new Error(`API request failed (${res.status}). Please retry.`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}/api${path}`, init);
  if (!res.ok) throw await failure(res);
  return res.json() as Promise<T>;
}

function post<T>(path: string, features: Features, signal?: AbortSignal) {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
    signal,
  });
}

let active: AbortController | undefined;
let sequence = 0;
let cfSequence = 0;
let explainController: AbortController | undefined;
let explainVersion = 0;
let featureTimer: ReturnType<typeof setTimeout> | undefined;

export function withDefaults(schema: Schema | null, features: Features): Features {
  const merged: Features = {};
  for (const feature of schema?.features ?? []) merged[feature.key] = feature.default;
  return { ...merged, ...features };
}

interface State {
  registry: Registry | null;
  schema: Schema | null;
  samples: Sample[];
  metrics: MetricsReport | null;
  analysis: Analysis | null;
  counterfactual: Counterfactual | null;
  features: Features;
  sampleId: string;
  selected: Vessel | null;
  mode: 'anatomy' | 'risk';
  loading: boolean;
  explaining: boolean;
  initializing: boolean;
  error: string | null;
  initError: string | null;
  initialize: () => Promise<void>;
  selectSample: (id: string) => Promise<void>;
  retry: () => Promise<void>;
  selectVessel: (v: Vessel | null) => void;
  setMode: (mode: 'anatomy' | 'risk') => void;
  setFeature: (key: string, value: string | number) => void;
  applyAdjustment: (updates: Features) => void;
}

type SetState = (partial: Partial<State>) => void;
type GetState = () => State;

async function loadCounterfactual(features: Features, target: string, set: SetState) {
  const version = ++cfSequence;
  try {
    const counterfactual = await post<Counterfactual>(`/counterfactual?target=${encodeURIComponent(target)}`, features);
    if (version === cfSequence) set({ counterfactual });
  } catch {
    if (version === cfSequence) set({ counterfactual: null });
  }
}

function startExplanations(features: Features, preferred: Target, set: SetState, get: GetState) {
  explainController?.abort();
  const version = ++explainVersion;
  const controller = new AbortController();
  explainController = controller;
  const order = [preferred, ...targets.filter(target => target !== preferred)];
  set({ explaining: true });
  void (async () => {
    try {
      for (const target of order) {
        if (version !== explainVersion) return;
        if (get().analysis?.explanations[target]) continue;
        const explained = await post<{ explanations: Analysis['explanations'] }>(`/explain?targets=${target}`, features, controller.signal);
        if (version !== explainVersion) return;
        const analysis = get().analysis;
        if (!analysis) return;
        set({ analysis: { ...analysis, explanations: { ...analysis.explanations, ...explained.explanations } } });
      }
      if (version === explainVersion) set({ explaining: false });
    } catch (e) {
      if (version === explainVersion && (e as Error).name !== 'AbortError') set({ explaining: false });
    }
  })();
}

async function score(features: Features, version: number, signal: AbortSignal, set: SetState, get: GetState) {
  const prediction = await post<Analysis['prediction']>('/predict', features, signal);
  if (version !== sequence) return;
  set({
    analysis: { prediction, explanations: {}, disclaimer: prediction.disclaimer },
    loading: false,
    mode: 'risk',
    error: null,
  });
  const target = get().selected ?? 'CAD';
  startExplanations(features, target, set, get);
  void loadCounterfactual(features, target, set);
}

/** Score the current record now, keeping the last estimate on screen until the new one lands. */
function runScore(set: SetState, get: GetState): Promise<void> {
  clearTimeout(featureTimer);
  active?.abort();
  explainController?.abort();
  const version = ++sequence;
  active = new AbortController();
  set({ loading: true, error: null });
  return score(get().features, version, active.signal, set, get).catch(e => {
    if (version === sequence && (e as Error).name !== 'AbortError') set({ loading: false, error: (e as Error).message });
  });
}

function scheduleScore(set: SetState, get: GetState) {
  clearTimeout(featureTimer);
  featureTimer = setTimeout(() => { void runScore(set, get); }, 250);
}

const roundTenth = (value: number) => Math.round(value * 10) / 10;

export const useStore = create<State>((set, get) => ({
  registry: null,
  schema: null,
  samples: [],
  metrics: null,
  analysis: null,
  counterfactual: null,
  features: {},
  sampleId: '',
  selected: null,
  mode: 'anatomy',
  loading: false,
  explaining: false,
  initializing: true,
  error: null,
  initError: null,
  initialize: async () => {
    set({ initializing: true, initError: null });
    try {
      const [registry, schema, samples, metrics] = await Promise.all([
        request<Registry>('/targets'),
        request<Schema>('/schema'),
        request<Sample[]>('/samples'),
        request<MetricsReport>('/metrics').catch(() => null),
      ]);
      set({
        registry,
        schema,
        samples,
        metrics,
        features: get().sampleId ? get().features : withDefaults(schema, {}),
        initializing: false,
      });
    } catch (e) {
      set({ initializing: false, initError: (e as Error).message });
    }
  },
  selectSample: async (id) => {
    clearTimeout(featureTimer);
    active?.abort();
    explainController?.abort();
    explainVersion += 1;
    cfSequence += 1;
    const version = ++sequence;
    active = new AbortController();
    const sample = id ? get().samples.find(s => s.id === id) : undefined;
    const features = withDefaults(get().schema, sample?.features ?? {});
    set({
      sampleId: id,
      features,
      analysis: null,
      counterfactual: null,
      explaining: false,
      error: null,
      loading: !!id,
      mode: 'anatomy',
    });
    if (!id) return;
    if (!sample) {
      set({ loading: false, error: 'Sample patient is unavailable.' });
      return;
    }
    try {
      await score(features, version, active.signal, set, get);
    } catch (e) {
      if (version === sequence && (e as Error).name !== 'AbortError') set({ loading: false, error: (e as Error).message });
    }
  },
  retry: () => runScore(set, get),
  selectVessel: selected => {
    set({ selected });
    const { analysis, features } = get();
    if (!analysis) return;
    const target = selected ?? 'CAD';
    void loadCounterfactual(features, target, set);
    if (!analysis.explanations[target]) startExplanations(features, target, set, get);
  },
  setMode: mode => set({ mode }),
  setFeature: (key, value) => {
    const features = { ...get().features, [key]: value };
    // Weight, height and BMI are one fact entered three ways; keep them consistent
    // (the counterfactual applies the same rule: a BMI change recomputes weight).
    const meters = Number(features.height) / 100;
    if (features.height != null && meters > 0) {
      if ((key === 'weight' || key === 'height') && features.weight != null) {
        features.bmi = roundTenth(Number(features.weight) / (meters * meters));
      } else if (key === 'bmi') {
        features.weight = roundTenth(Number(value) * meters * meters);
      }
    }
    set({ features });
    scheduleScore(set, get);
  },
  applyAdjustment: updates => {
    const features = { ...get().features };
    for (const [key, value] of Object.entries(updates)) {
      if (typeof value === 'string' || typeof value === 'number') features[key] = value;
    }
    set({ features });
    scheduleScore(set, get);
  },
}));
