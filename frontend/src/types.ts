export type Vessel = 'LAD' | 'LCX' | 'RCA';
export type Target = Vessel | 'CAD';
export type Features = Record<string, string | number | null>;
export interface TargetSpec { label: string; kind: 'overall' | 'vessel'; meshes?: string[]; description?: string; territory?: string }
export interface Registry { version: number; targets: Record<Target, TargetSpec> }
export interface Schema { features: { key: string; default: string | number }[] }
export interface Sample { id: string; name: string; description: string; features: Features }
export interface Prediction { target: string; label: string; probability: number; predicted_positive: boolean; threshold: number; ci: [number, number] | null }
export interface Contribution { key: string; label: string; category: string; value: unknown; unit: string | null; out_of_range: boolean | null; contribution: number }
export interface Explanation { target: string; method: string; base_value: number; output_value: number; probability: number; contributions: Contribution[]; category_totals: Record<string, number> }
export interface Analysis {
  prediction: {
    model: { name: string; version: string; is_placeholder: boolean };
    cad: Prediction; vessels: Record<Vessel, Prediction>; imputed: string[]; disclaimer: string;
  };
  explanations: Partial<Record<Target, Explanation>>;
  disclaimer: string;
}
