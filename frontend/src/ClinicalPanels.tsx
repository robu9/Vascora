import { useState } from 'react';
import { percent } from './mapping';
import { useStore } from './store';
import type { Contribution, Features, Target } from './types';

const physiology = new Set(['vitals', 'labs', 'echo']);
const shown = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] as const;
const metricLabel: Record<string, string> = {
  accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1', roc_auc: 'ROC-AUC',
};

function formatMetric(key: string, value: number) {
  return key === 'roc_auc' || key === 'f1' ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`;
}

function formatValue(feature: Contribution) {
  if (typeof feature.value === 'number') {
    const text = Number.isInteger(feature.value) ? String(feature.value) : feature.value.toFixed(1);
    return feature.unit ? `${text} ${feature.unit}` : text;
  }
  return feature.unit ? `${String(feature.value)} ${feature.unit}` : String(feature.value);
}

export function ExplanationPanel({ target }: { target: Target }) {
  const schema = useStore(s => s.schema);
  const analysis = useStore(s => s.analysis);
  const explaining = useStore(s => s.explaining);
  const explanation = analysis?.explanations[target];
  const counterfactual = useStore(s => s.counterfactual);
  const applyAdjustment = useStore(s => s.applyAdjustment);
  if (!analysis) return null;
  if (!explanation) {
    return <section className="explain-section" aria-label="Why this estimate">
      <div className="section-title"><span className="eyebrow">WHY THIS ESTIMATE</span><span className="small-tag">SHAP</span></div>
      <p role="status">{explaining ? `Reading the factors behind ${target}…` : 'Factor breakdown is unavailable for this estimate.'}</p>
    </section>;
  }
  const categories = schema?.categories ?? {};
  const totals = Object.entries(explanation.category_totals).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxTotal = Math.max(...totals.map(([, value]) => Math.abs(value)), 0.001);
  const measurements = explanation.contributions.filter(feature => physiology.has(feature.category));
  const maxContribution = Math.max(...explanation.contributions.map(feature => Math.abs(feature.contribution)), 0.001);
  const hints = counterfactual?.target === target ? counterfactual : null;
  return <section className="explain-section" aria-label="Why this estimate">
    <div className="section-title"><span className="eyebrow">WHY THIS ESTIMATE</span><span className="small-tag">SHAP</span></div>
    <p className="method">{explanation.method}. Values are log-odds added to a base of {explanation.base_value.toFixed(2)}.</p>
    <h3>Contribution by clinical group</h3>
    <ul className="category-bars">
      {totals.map(([key, value]) => <li key={key}>
        <span>{categories[key] ?? key}</span>
        <span className="bar-track" aria-hidden="true"><span className={value >= 0 ? 'bar-up' : 'bar-down'} style={{ width: `${(Math.abs(value) / maxTotal) * 100}%` }}/></span>
        <span className={value > 0 ? 'increase' : 'decrease'}>{value > 0 ? '+' : ''}{value.toFixed(2)}</span>
      </li>)}
    </ul>
    <h3>Strongest factors</h3>
    <ol className="waterfall">
      {explanation.contributions.slice(0, 8).map(feature => <li key={feature.key}>
        <span>{feature.label}<small>{formatValue(feature)}{feature.out_of_range ? ' · outside typical range' : ''}</small></span>
        <span className="bar-track" aria-hidden="true"><span className={feature.contribution >= 0 ? 'bar-up' : 'bar-down'} style={{ width: `${(Math.abs(feature.contribution) / maxContribution) * 100}%` }}/></span>
        <span className={feature.contribution > 0 ? 'increase' : 'decrease'}>{feature.contribution > 0 ? '+' : ''}{feature.contribution.toFixed(3)}</span>
      </li>)}
    </ol>
    <h3>Physiological measurements</h3>
    <p className="fine-print">Each measurement is shown with its contribution to the {target} estimate. Direction is in log-odds, not percentage points.</p>
    <div className="measure-table" role="table" aria-label="Measurements and contributions">
      <div className="measure-head" role="row"><span>Measurement</span><span>Value</span><span>Effect</span></div>
      {measurements.map(feature => <div className="measure-row" role="row" key={feature.key}>
        <span>{feature.label}{feature.out_of_range ? <em> outside range</em> : null}</span>
        <span>{formatValue(feature)}</span>
        <span className={feature.contribution > 0 ? 'increase' : 'decrease'}>{feature.contribution > 0 ? '↑' : feature.contribution < 0 ? '↓' : '—'} {Math.abs(feature.contribution).toFixed(3)}</span>
      </div>)}
    </div>
    <h3>What the model scores as lower risk</h3>
    {hints?.changes.length ? <ol className="hints">
      {hints.changes.slice(0, 4).map(change => <li key={change.key}>
        <span>{change.label}<small>{String(change.from_value)} → {String(change.to_value)}{change.unit ? ` ${change.unit}` : ''}</small></span>
        <span className="hint-actions"><span className="decrease">{percent(change.probability)}</span><button type="button" className="text-button" onClick={() => {
          const updates: Features = {};
          if (typeof change.to_value === 'string' || typeof change.to_value === 'number') updates[change.key] = change.to_value;
          for (const [key, value] of Object.entries(change.coupled)) {
            if (typeof value === 'string' || typeof value === 'number') updates[key] = value;
          }
          applyAdjustment(updates);
        }}>Apply</button></span>
      </li>)}
      {hints.combined.length > 1 && <li className="combined"><span>Combined<small>{hints.combined.length} changes together</small></span><span className="decrease">{percent(hints.combined_probability)}</span></li>}
    </ol> : <p className="fine-print">{hints?.note ?? 'Hints appear after an estimate is ready. Only changes the trained model actually scores as lower risk are listed.'}</p>}
  </section>;
}

export function MetricsPanel() {
  const metrics = useStore(s => s.metrics);
  const [open, setOpen] = useState(false);
  const targets = metrics?.targets ? Object.keys(metrics.targets) : [];
  return <section className="metrics-section">
    <div className="section-title">
      <span className="eyebrow">HELD-OUT EVALUATION</span>
      <button className="text-button" aria-expanded={open} onClick={() => setOpen(value => !value)}>{open ? 'Hide' : 'Show'} metrics</button>
    </div>
    {metrics?.available === false && <p className="fine-print">{metrics.detail}</p>}
    {metrics?.dataset && <p className="fine-print">{metrics.dataset.name}. {metrics.dataset.n_total} patients, {metrics.dataset.n_features} inputs, {metrics.dataset.n_test} held out.</p>}
    {targets.length > 0 && <p className="auc-line">{targets.map(target => {
      const auc = metrics?.targets?.[target]?.test?.metrics?.roc_auc;
      return auc == null ? null : <span key={target}>{target} <strong>{auc.toFixed(3)}</strong></span>;
    })}</p>}
    {open && targets.length > 0 && <div className="metrics-table" role="table" aria-label="Classification metrics">
      <div className="measure-head" role="row"><span>Target</span>{shown.map(key => <span key={key}>{metricLabel[key]}</span>)}</div>
      {targets.map(target => {
        const row = metrics?.targets?.[target]?.test?.metrics ?? {};
        const cv = metrics?.targets?.[target]?.cv?.ensemble?.roc_auc;
        return <div className="measure-row" role="row" key={target}>
          <span>{target}{cv ? <small>CV AUC {cv.mean.toFixed(3)} ± {cv.std.toFixed(3)}</small> : null}</span>
          {shown.map(key => <span key={key}>{row[key] == null ? '—' : formatMetric(key, row[key])}</span>)}
        </div>;
      })}
    </div>}
    {open && metrics?.validation && <p className="fine-print">{metrics.validation.scheme}</p>}
  </section>;
}

const disclaimer = 'For decision support and educational purposes only. Not a substitute for coronary angiography or formal diagnostic imaging.';

export function DisclaimerModal() {
  const [open, setOpen] = useState(() => sessionStorage.getItem('vascora-disclaimer') !== '1');
  if (!open) return null;
  return <div className="modal-backdrop">
    <div className="modal" role="dialog" aria-modal="true" aria-labelledby="disclaimer-title">
      <span className="eyebrow">CLINICAL SAFETY</span>
      <h2 id="disclaimer-title">Decision support only</h2>
      <p>{disclaimer}</p>
      <p>Colors on LAD, LCX and RCA show predicted stenosis probability. They do not locate plaque, measure narrowing, or replace formal diagnostic imaging.</p>
      <button onClick={() => { sessionStorage.setItem('vascora-disclaimer', '1'); setOpen(false); }}>I understand</button>
    </div>
  </div>;
}
