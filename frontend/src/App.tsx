import { useEffect } from 'react';
import { HeartScene } from './HeartScene';
import { useStore } from './store';
import { percent, vessels } from './mapping';
const disclaimer = 'For decision support and educational purposes only. Not a substitute for coronary angiography or formal diagnostic imaging.';
export default function App() {
  const state = useStore();
  useEffect(() => { void state.initialize(); }, []);
  const { analysis, selected, registry } = state;
  const target = selected ?? 'CAD';
  const prediction = selected ? analysis?.prediction.vessels[selected] : analysis?.prediction.cad;
  const explanation = analysis?.explanations[target];
  const sample = state.samples.find(s => s.id === state.sampleId);
  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="Vascora home"><span className="brand-icon">v</span>vascora<span className="brand-dot">.</span></a><span className="header-divider"/><span className="product-label">CORONARY INTELLIGENCE</span><span className="research-tag">EDUCATIONAL PREVIEW</span></header>
    <div className="disclaimer"><span aria-hidden="true">ⓘ</span><span>{analysis?.disclaimer ?? disclaimer}</span></div>
    <main><HeartScene/><aside className="results" aria-label="Patient and risk results">
      <section className="patient-section"><span className="eyebrow">PATIENT PROFILE</span><h2>Explore a sample</h2><p>Choose a preset to see the model’s estimates on the heart.</p>
        <label className="sr-only" htmlFor="patient">Sample patient</label><select id="patient" value={state.sampleId} disabled={state.initializing || !!state.initError} onChange={e => void state.selectSample(e.target.value)}><option value="">Select a sample patient</option>{state.samples.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select>
        {sample && <p className="sample-description">{sample.description}</p>}
        {state.initializing && <p role="status">Connecting to the API…</p>}
        {state.initError && <div role="alert" className="error">{state.initError}<button onClick={() => void state.initialize()}>Retry connection</button></div>}
        {state.loading && <p role="status">Analyzing sample…</p>}
        {state.error && <div role="alert" className="error">{state.error}<button onClick={() => void state.selectSample(state.sampleId)}>Retry analysis</button></div>}
      </section>
      {analysis?.prediction.model.is_placeholder && <div className="placeholder-tag">DEMO MODEL · HEURISTIC PLACEHOLDER<p>These estimates are not from a trained model.</p></div>}
      <section className="risk-summary"><div className="section-title"><span className="eyebrow">OVERALL CAD ESTIMATE</span><span className="small-tag">CAD</span></div><div className="big-probability">{analysis ? percent(analysis.prediction.cad.probability) : '—'}<span>estimated probability</span></div>{analysis && <p className="classification">{analysis.prediction.cad.predicted_positive ? 'Above' : 'Below'} model threshold · {percent(analysis.prediction.cad.threshold)}</p>}</section>
      <section className="vessel-section"><div className="section-title"><span className="eyebrow">CORONARY ARTERIES</span><span className="hint">Select to explore</span></div><div className="vessel-list">{vessels.map(v => <button className={`vessel-card ${selected === v ? 'selected' : ''}`} key={v} aria-pressed={selected === v} onClick={() => state.selectVessel(v)}><span className="vessel-symbol">{v === 'LAD' ? '01' : v === 'LCX' ? '02' : '03'}</span><span className="vessel-name"><strong>{v}</strong><small>{registry?.targets[v]?.label ?? v}</small></span><span className="vessel-value">{analysis ? percent(analysis.prediction.vessels[v].probability) : '—'}<small>↗</small></span></button>)}</div>
      <div className="legend"><div className="legend-gradient"/><div><span>0% · lower probability</span><span>100% · higher</span></div></div></section>
      <section className="detail-section" aria-live="polite"><div className="section-title"><span className="eyebrow">{target} · DETAILS</span>{selected && <button className="text-button" onClick={() => state.selectVessel(null)}>View CAD</button>}</div><p>{selected ? registry?.targets[selected]?.description : 'Select a vessel on the heart or in the list to inspect its estimate.'}</p>
        {prediction && <dl className="details-grid"><div><dt>Probability</dt><dd>{percent(prediction.probability)}</dd></div><div><dt>Threshold</dt><dd>{percent(prediction.threshold)}</dd></div><div><dt>Classification</dt><dd>{prediction.predicted_positive ? 'Above threshold' : 'Below threshold'}</dd></div><div><dt>Confidence interval</dt><dd>{prediction.ci ? `${percent(prediction.ci[0])}–${percent(prediction.ci[1])}` : 'Not available'}</dd></div></dl>}
        {explanation && <><h3>Top contributing factors</h3><p className="method">{explanation.method}</p><ol className="factors">{[...explanation.contributions].sort((a,b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0,5).map(f => <li key={f.key}><span>{f.label}<small>{String(f.value)} {f.unit ?? ''}</small></span><span className={f.contribution > 0 ? 'increase' : 'decrease'}>{f.contribution > 0 ? '↑' : f.contribution < 0 ? '↓' : '—'} {Math.abs(f.contribution).toFixed(3)}</span></li>)}</ol><p className="fine-print">Contributions are in log-odds, not percentage points. Arrows indicate direction.</p></>}
      </section>
      <footer className="results-footer">Representative anatomy, not patient imaging. Colors show vessel-level probability, not blockage location or severity.{analysis && <p>{analysis.prediction.imputed.length} inputs filled with schema defaults · Model {analysis.prediction.model.version}</p>}<a href="/models/ATTRIBUTION.md" target="_blank" rel="noreferrer">Anatomy credits & license ↗</a></footer>
    </aside></main><footer className="page-footer"><span>VASCORA / ANATOMY EXPLORER</span><span>Built for understanding.</span></footer>
  </div>;
}
