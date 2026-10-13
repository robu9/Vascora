import { useStore } from './store';
import type { FeatureSpec } from './types';

const closed = new Set(['labs']);

function stepFor(spec: FeatureSpec) {
  return (spec.max ?? 100) <= 80 ? 0.1 : 1;
}

function labelFor(spec: FeatureSpec, value: string | number) {
  return spec.option_labels?.[String(value)] ?? String(value);
}

function Field({ spec }: { spec: FeatureSpec }) {
  const value = useStore(s => s.features[spec.key] ?? spec.default);
  const setFeature = useStore(s => s.setFeature);
  const id = `feature-${spec.key}`;
  if (spec.type === 'number') {
    const numeric = Number(value);
    return <div className="field">
      <div className="field-label"><label htmlFor={id}>{spec.label}</label>{spec.unit ? <small>{spec.unit}</small> : null}</div>
      <input id={id} type="number" min={spec.min} max={spec.max} step={stepFor(spec)} value={Number.isFinite(numeric) ? numeric : ''}
        onChange={e => { if (e.target.value !== '') setFeature(spec.key, Number(e.target.value)); }}/>
    </div>;
  }
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
  const state = useStore();
  const categories = state.schema?.categories ?? {};
  const grouped = Object.keys(categories).map(key => ({
    key,
    label: categories[key],
    features: state.schema?.features.filter(feature => feature.category === key) ?? [],
  })).filter(group => group.features.length);
  return <aside className="intake" aria-label="Clinical inputs">
    <span className="eyebrow">CLINICAL INPUTS</span>
    <h2>Patient data</h2>
    <p>Presets load a full record. Changing any value re-scores CAD and the three arteries, then recolors the heart.</p>
    <label className="sr-only" htmlFor="patient">Sample patient</label>
    <select id="patient" value={state.sampleId} disabled={state.initializing || !!state.initError} onChange={e => void state.selectSample(e.target.value)}>
      <option value="">Select a sample patient</option>
      {state.samples.map(sample => <option key={sample.id} value={sample.id}>{sample.name}</option>)}
    </select>
    {state.samples.find(sample => sample.id === state.sampleId)?.description && <p className="sample-description">{state.samples.find(sample => sample.id === state.sampleId)?.description}</p>}
    {state.initializing && <p role="status">Connecting to the API…</p>}
    {state.initError && <div role="alert" className="error">{state.initError}<button onClick={() => void state.initialize()}>Retry connection</button></div>}
    {state.loading && <p role="status">Analyzing patient…</p>}
    {state.error && <div role="alert" className="error">{state.error}<button onClick={() => void state.selectSample(state.sampleId)}>Retry analysis</button></div>}
    {grouped.map(group => <details key={group.key} open={!closed.has(group.key)}>
      <summary>{group.label}</summary>
      {group.key === 'echo' && <p className="fine-print">Region with RWMA is an echocardiographic input. It is not a 3D lesion map.</p>}
      <div className="fields">{group.features.map(feature => <Field key={feature.key} spec={feature}/>)}</div>
    </details>)}
    <p className="fine-print">LAD, LCX, RCA and angiography status are never used as inputs.</p>
  </aside>;
}
