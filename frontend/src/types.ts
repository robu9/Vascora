export type Vessel = 'LAD' | 'LCX' | 'RCA';
export type Target = Vessel | 'CAD';
export type Features = Record<string, string | number | null>;

export interface TargetSpec {
  label: string;
  kind: 'overall' | 'vessel';
  meshes?: string[];
  description?: string;
  territory?: string;
}

export interface Registry { version: number; targets: Record<Target, TargetSpec> }

export interface FeatureSpec {
  key: string;
  label: string;
  type: 'number' | 'binary' | 'yn' | 'categorical' | 'ordinal';
  unit?: string | null;
  min?: number;
  max?: number;
  normal?: [number, number] | null;
  category: string;
  modifiable?: boolean;
  default: string | number;
  options?: (string | number)[];
  option_labels?: Record<string, string>;
}

export interface Schema {
  categories: Record<string, string>;
  features: FeatureSpec[];
}

export interface Sample { id: string; name: string; description: string; features: Features }

export interface Prediction {
  target: string;
  label: string;
  probability: number;
  predicted_positive: boolean;
  threshold: number;
  ci: [number, number] | null;
}

export interface Contribution {
  key: string;
  label: string;
  category: string;
  value: unknown;
  unit: string | null;
  out_of_range: boolean | null;
  contribution: number;
}

export interface Explanation {
  target: string;
  method: string;
  base_value: number;
  output_value: number;
  probability: number;
  contributions: Contribution[];
  category_totals: Record<string, number>;
}

export interface Analysis {
  prediction: {
    model: { name: string; version: string; is_placeholder: boolean };
    cad: Prediction;
    vessels: Record<Vessel, Prediction>;
    imputed: string[];
    disclaimer: string;
  };
  explanations: Partial<Record<Target, Explanation>>;
  disclaimer: string;
}

export interface CounterfactualChange {
  key: string;
  label: string;
  category: string;
  unit: string | null;
  from_value: unknown;
  to_value: unknown;
  coupled: Record<string, unknown>;
  probability: number;
  delta: number;
}

export interface Counterfactual {
  target: string;
  baseline_probability: number;
  changes: CounterfactualChange[];
  combined: CounterfactualChange[];
  combined_probability: number;
  note: string | null;
  disclaimer: string;
}

export interface MetricStats { mean: number; std: number }

export interface MetricsReport {
  available?: boolean;
  detail?: string;
  dataset?: { name: string; n_total: number; n_dev: number; n_test: number; n_features: number };
  validation?: { scheme: string };
  targets?: Record<string, {
    test?: { metrics?: Record<string, number>; ci95?: Record<string, [number, number]> };
    cv?: { ensemble?: Record<string, MetricStats> };
  }>;
}
