import { useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useStore } from './store';
import type { FeatureSpec } from './types';

const closed = new Set(['labs']);

function stepFor(spec: FeatureSpec) {
  return (spec.max ?? 100) <= 80 ? 0.1 : 1;
}

function labelFor(spec: FeatureSpec, value: string | number) {
  return spec.option_labels?.[String(value)] ?? String(value);
}

const parse = (text: string) => (text.trim() === '' ? Number.NaN : Number(text));

function inRange(spec: FeatureSpec, value: number) {
  return Number.isFinite(value) && (spec.min == null || value >= spec.min) && (spec.max == null || value <= spec.max);
}

/** Keeps the typed text locally so empty or partial entries are never sent for scoring. */
function NumberField({ spec }: { spec: FeatureSpec }) {
  const stored = useStore(s => s.features[spec.key] ?? spec.default);
  const setFeature = useStore(s => s.setFeature);
  const [draft, setDraft] = useState(String(stored));
  const [synced, setSynced] = useState(stored);
  if (stored !== synced) {
    // The store changed elsewhere (preset, BMI coupling, applied hint). Show it, unless it is
    // simply the number being typed, so "1." is not rewritten to "1" mid-entry.
    setSynced(stored);
    if (parse(draft) !== Number(stored)) setDraft(String(stored));
  }
  const id = `feature-${spec.key}`;
  const invalid = !inRange(spec, parse(draft));
  const range = spec.min != null && spec.max != null ? `${spec.min}–${spec.max}` : 'a number';
  return <div className="field">
    <div className="field-label"><label htmlFor={id}>{spec.label}</label>{spec.unit ? <small>{spec.unit}</small> : null}</div>
    <input id={id} type="number" min={spec.min} max={spec.max} step={stepFor(spec)} value={draft}
      aria-invalid={invalid} aria-describedby={invalid ? `${id}-hint` : undefined}
      onChange={e => {
        setDraft(e.target.value);
        const next = parse(e.target.value);
        if (inRange(spec, next) && next !== Number(stored)) setFeature(spec.key, next);
      }}
      onBlur={() => { if (invalid) setDraft(String(stored)); }}/>
    {invalid && <small id={`${id}-hint`} className="field-hint">Enter {range}{spec.unit ? ` ${spec.unit}` : ''}. The estimate still uses {String(stored)}.</small>}
  </div>;
}

function ChoiceField({ spec }: { spec: FeatureSpec }) {
  const value = useStore(s => s.features[spec.key] ?? spec.default);
  const setFeature = useStore(s => s.setFeature);
  const id = `feature-${spec.key}`;
  if (spec.type === 'binary' || spec.type === 'yn') {
    const on = spec.type === 'binary' ? 1 : 'Y';
    const off = spec.type === 'binary' ? 0 : 'N';
    return <div className="field field-toggle">
      <span id={id}>{spec.label}</span>
      <div role="group" aria-labelledby={id}>
        <button type="button" aria-pressed={value === on} onClick={() => setFeature(spec.key, on)}>Yes</button>
        <button type="button" aria-pressed={value === off} onClick={() => setFeature(spec.key, off)}>No</button>
      </div>
    </div>;
  }
  return <div className="field">
    <label htmlFor={id}>{spec.label}</label>
    <select id={id} value={String(value)} onChange={e => {
      const next = spec.options?.find(option => String(option) === e.target.value) ?? e.target.value;
      setFeature(spec.key, next);
    }}>
      {(spec.options ?? []).map(option => <option key={String(option)} value={String(option)}>{labelFor(spec, option)}</option>)}
    </select>
  </div>;
}

export function PatientForm() {
  const { schema, samples, sampleId, initializing, initError, loading, error } = useStore(useShallow(s => ({
    schema: s.schema, samples: s.samples, sampleId: s.sampleId, initializing: s.initializing,
    initError: s.initError, loading: s.loading, error: s.error,
  })));
  const selectSample = useStore(s => s.selectSample);
  const initialize = useStore(s => s.initialize);
  const retry = useStore(s => s.retry);
  // On phones the form sits between the heart and the results; start with groups folded.
  const [compact] = useState(() => matchMedia('(max-width: 760px)').matches);
  const categories = schema?.categories ?? {};
  const grouped = Object.keys(categories).map(key => ({
    key,
    label: categories[key],
    features: schema?.features.filter(feature => feature.category === key) ?? [],
  })).filter(group => group.features.length);
  const description = samples.find(sample => sample.id === sampleId)?.description;
  return <aside className="intake" aria-label="Clinical inputs">
    <span className="eyebrow">CLINICAL INPUTS</span>
    <h2>Patient data</h2>
    <p>Presets load a full record. Changing any value re-scores CAD and the three arteries, then recolors the heart.</p>
    <label className="sr-only" htmlFor="patient">Sample patient</label>
    <select id="patient" value={sampleId} disabled={initializing || !!initError} onChange={e => void selectSample(e.target.value)}>
      <option value="">Select a sample patient</option>
      {samples.map(sample => <option key={sample.id} value={sample.id}>{sample.name}</option>)}
    </select>
    {description && <p className="sample-description">{description}</p>}
    {initializing && <p role="status">Connecting to the API…</p>}
    {initError && <div role="alert" className="error">{initError}<button onClick={() => void initialize()}>Retry connection</button></div>}
    {loading && <p role="status">Analyzing patient…</p>}
    {error && <div role="alert" className="error">{error}<button onClick={() => void retry()}>Retry analysis</button></div>}
    {grouped.map(group => <details key={group.key} open={!compact && !closed.has(group.key)}>
      <summary>{group.label}</summary>
      {group.key === 'echo' && <p className="fine-print">Region with RWMA is an echocardiographic input. It is not a 3D lesion map.</p>}
      <div className="fields">{group.features.map(feature => feature.type === 'number'
        ? <NumberField key={feature.key} spec={feature}/>
        : <ChoiceField key={feature.key} spec={feature}/>)}</div>
    </details>)}
    <p className="fine-print">LAD, LCX, RCA and angiography status are never used as inputs.</p>
  </aside>;
}
