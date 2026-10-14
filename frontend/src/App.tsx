import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { DisclaimerModal, ExplanationPanel, MetricsPanel } from './ClinicalPanels';
import { HeartScene } from './HeartScene';
import { PatientForm } from './PatientForm';
import { percent, percentRange, riskColor, riskGradient, vessels } from './mapping';
import { useStore } from './store';

const disclaimer = 'For decision support and educational purposes only. Not a substitute for coronary angiography or formal diagnostic imaging.';

export default function App() {
  const initialize = useStore(s => s.initialize);
  const selectVessel = useStore(s => s.selectVessel);
  const { analysis, selected, registry, loading } = useStore(useShallow(s => ({
    analysis: s.analysis, selected: s.selected, registry: s.registry, loading: s.loading,
  })));
  useEffect(() => { void initialize(); }, [initialize]);
  const target = selected ?? 'CAD';
  const prediction = selected ? analysis?.prediction.vessels[selected] : analysis?.prediction.cad;
  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="Vascora home"><span className="brand-icon">v</span>vascora<span className="brand-dot">.</span></a>
      <span className="header-divider"/>
      <span className="product-label">CORONARY INTELLIGENCE</span>
      <span className="research-tag">EDUCATIONAL PREVIEW</span>
    </header>
    <div className="disclaimer"><span aria-hidden="true">ⓘ</span><span>{analysis?.disclaimer ?? disclaimer}</span></div>
    <main>
      <HeartScene/>
      <aside className="results" aria-label="Patient and risk results" aria-busy={loading && !!analysis}>
        {analysis?.prediction.model.is_placeholder && <div className="placeholder-tag">DEMO MODEL · HEURISTIC PLACEHOLDER<p>These estimates are not from a trained model.</p></div>}
        <section className="risk-summary">
          <div className="section-title"><span className="eyebrow">OVERALL CAD ESTIMATE</span><span className="small-tag">CAD</span></div>
          <div className="big-probability">{analysis ? percent(analysis.prediction.cad.probability) : '—'}<span>estimated probability</span></div>
          {loading && !analysis && <p className="classification" role="status">Scoring LAD, LCX and RCA…</p>}
          {!loading && !analysis && <p className="empty-state">Choose a sample patient, or change any clinical input, to estimate risk.</p>}
          {analysis && <p className="classification">{analysis.prediction.cad.predicted_positive ? 'Above' : 'Below'} model threshold · {percent(analysis.prediction.cad.threshold)}</p>}
        </section>
        <section className="vessel-section">
          <div className="section-title"><span className="eyebrow">CORONARY ARTERIES</span><span className="hint">Select to explore</span></div>
          <div className="vessel-list">{vessels.map(vessel => {
            const estimate = analysis?.prediction.vessels[vessel];
            return <button className={`vessel-card ${selected === vessel ? 'selected' : ''}`} key={vessel} aria-pressed={selected === vessel} onClick={() => selectVessel(vessel)}>
              <span className="vessel-symbol">{vessel === 'LAD' ? '01' : vessel === 'LCX' ? '02' : '03'}</span>
              <span className="vessel-name"><strong>{vessel}</strong><small>{registry?.targets[vessel]?.label ?? vessel}</small></span>
              <span className="vessel-value">{estimate ? percent(estimate.probability) : '—'}</span>
              {estimate && <span className="vessel-bar" aria-hidden="true">
                <span style={{ width: `${estimate.probability * 100}%`, background: riskColor(estimate.probability) }}/>
                <i className="threshold-tick" style={{ left: `${estimate.threshold * 100}%` }}/>
              </span>}
            </button>;
          })}</div>
          <div className="legend"><div className="legend-gradient" style={{ background: riskGradient }}/><div><span>Lower probability</span><span><i className="threshold-tick legend-tick" aria-hidden="true"/> model threshold</span><span>Higher</span></div></div>
        </section>
        <section className="detail-section" aria-live="polite">
          <div className="section-title"><span className="eyebrow">{target} · DETAILS</span>{selected && <button className="text-button" onClick={() => selectVessel(null)}>View CAD</button>}</div>
          <p>{selected ? registry?.targets[selected]?.description : 'Select a vessel on the heart or in the list to inspect its estimate and the factors behind it.'}</p>
          {prediction && <dl className="details-grid">
            <div><dt>Probability</dt><dd>{percent(prediction.probability)}</dd></div>
            <div><dt>Threshold</dt><dd>{percent(prediction.threshold)}</dd></div>
            <div><dt>Classification</dt><dd>{prediction.predicted_positive ? 'Above threshold' : 'Below threshold'}</dd></div>
            <div><dt>90% confidence interval</dt><dd>{prediction.ci ? percentRange(prediction.ci[0], prediction.ci[1]) : 'Not available'}</dd></div>
          </dl>}
        </section>
        <ExplanationPanel target={target}/>
        <MetricsPanel/>
        <footer className="results-footer">
          Representative anatomy, not patient imaging. Colors show vessel-level probability, not blockage location or severity.
          {analysis && <p>{analysis.prediction.imputed.length} inputs filled with schema defaults · Model {analysis.prediction.model.version}</p>}
          <a href="/models/ATTRIBUTION.md" target="_blank" rel="noreferrer">Anatomy credits & license ↗</a>
        </footer>
      </aside>
      <PatientForm/>
    </main>
    <footer className="page-footer"><span>VASCORA / ANATOMY EXPLORER</span><span>Built for understanding.</span></footer>
    <DisclaimerModal/>
  </div>;
}
