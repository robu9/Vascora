import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useStore } from './store';
import type { Analysis } from './types';
const sample = (id: string) => ({ id, name: id, description: '', features: { age: 60 } });
const scored = (name: string) => ({ model: { name }, disclaimer: '', imputed: [] });
const answer = (name: string) => ({ prediction: scored(name), explanations: {}, disclaimer: '' }) as unknown as Analysis;
beforeEach(() => {
  vi.useRealTimers();
  useStore.setState({
    samples: [sample('a'), sample('b')], analysis: null, counterfactual: null, features: {},
    schema: null, loading: false, explaining: false, error: null, mode: 'anatomy', sampleId: '',
  });
});
afterEach(() => { vi.clearAllTimers(); vi.unstubAllGlobals(); vi.useRealTimers(); });
it('clears old results immediately when changing patients', async () => {
  useStore.setState({ analysis: answer('old') });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(scored('new')))));
  const pending = useStore.getState().selectSample('a');
  expect(useStore.getState().analysis).toBeNull(); expect(useStore.getState().mode).toBe('anatomy');
  await pending; expect(useStore.getState().analysis?.prediction.model.name).toBe('new');
});
it('rejects stale responses even when transport ignores cancellation', async () => {
  const resolvers: ((r: Response) => void)[] = [];
  const signals: AbortSignal[] = [];
  vi.stubGlobal('fetch', vi.fn((_url, init) => { signals.push(init.signal); return new Promise<Response>(resolve => resolvers.push(resolve)); }));
  const first = useStore.getState().selectSample('a'); const second = useStore.getState().selectSample('b');
  expect(signals[0].aborted).toBe(true);
  resolvers[1](new Response(JSON.stringify(scored('b')))); await second;
  resolvers[0](new Response(JSON.stringify(scored('a')))); await first;
  expect(useStore.getState().analysis?.prediction.model.name).toBe('b');
});
it('leaves neutral anatomy available after failure and supports retry', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('', { status: 503 })).mockResolvedValue(new Response(JSON.stringify(scored('retry')))));
  await useStore.getState().selectSample('a');
  expect(useStore.getState().error).toContain('503'); expect(useStore.getState().analysis).toBeNull();
  await useStore.getState().selectSample('a'); expect(useStore.getState().error).toBeNull();
});
it('re-scores an edited feature without dropping the current estimate first', async () => {
  vi.useFakeTimers();
  useStore.setState({ analysis: answer('kept'), features: { age: 50 }, mode: 'risk' });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(scored('edited')))));
  useStore.getState().setFeature('age', 80);
  expect(useStore.getState().features.age).toBe(80);
  expect(useStore.getState().analysis?.prediction.model.name).toBe('kept');
  await vi.advanceTimersByTimeAsync(250);
  expect(useStore.getState().analysis?.prediction.model.name).toBe('edited');
  expect(useStore.getState().mode).toBe('risk');
});
it('keeps body mass consistent when height, weight or BMI changes', () => {
  vi.useFakeTimers();
  vi.stubGlobal('fetch', vi.fn());
  useStore.setState({ features: { weight: 80, height: 180, bmi: 20 } });
  useStore.getState().setFeature('weight', 90);
  expect(useStore.getState().features.bmi).toBe(27.8);
  useStore.getState().setFeature('bmi', 25);
  expect(useStore.getState().features.weight).toBe(81);
});
it('retries the edited record instead of reloading the preset', async () => {
  vi.useFakeTimers();
  useStore.setState({ sampleId: 'a', features: { age: 60 } });
  const fetch = vi.fn()
    .mockResolvedValueOnce(new Response('', { status: 503 }))
    .mockImplementation(() => Promise.resolve(new Response(JSON.stringify(scored('retried')))));
  vi.stubGlobal('fetch', fetch);
  useStore.getState().setFeature('age', 81);
  await vi.advanceTimersByTimeAsync(250);
  await vi.waitFor(() => expect(useStore.getState().error).toContain('503'));
  await useStore.getState().retry();
  expect(useStore.getState().error).toBeNull();
  expect(useStore.getState().features.age).toBe(81);
  expect(JSON.parse(fetch.mock.calls[1][1].body).age).toBe(81);
  expect(useStore.getState().analysis?.prediction.model.name).toBe('retried');
});
it('shows the field and reason when the API rejects an input', async () => {
  const detail = [{ loc: ['body', 'age'], msg: 'Input should be greater than or equal to 18' }];
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify({ detail }), { status: 422 }))));
  await useStore.getState().selectSample('a');
  expect(useStore.getState().error).toBe('age: Input should be greater than or equal to 18');
});
it('colors the heart before the factor breakdown returns', async () => {
  const pending: ((r: Response) => void)[] = [];
  vi.stubGlobal('fetch', vi.fn((url: string) => {
    if (String(url).includes('/explain')) return new Promise<Response>(resolve => pending.push(resolve));
    return Promise.resolve(new Response(JSON.stringify(scored('fast'))));
  }));
  await useStore.getState().selectSample('a');
  expect(useStore.getState().mode).toBe('risk');
  expect(useStore.getState().loading).toBe(false);
  expect(useStore.getState().analysis?.prediction.model.name).toBe('fast');
  expect(useStore.getState().analysis?.explanations.CAD).toBeUndefined();
  expect(useStore.getState().explaining).toBe(true);
  pending[0](new Response(JSON.stringify({ explanations: { CAD: { target: 'CAD' } } })));
  await vi.waitFor(() => expect(useStore.getState().analysis?.explanations.CAD?.target).toBe('CAD'));
});
it('returning to anatomy cancels pending analysis', async () => {
  let resolve!: (r: Response) => void;
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(r => { resolve = r; })));
  const first = useStore.getState().selectSample('a');
  await useStore.getState().selectSample(''); resolve(new Response(JSON.stringify(scored('a')))); await first;
  expect(useStore.getState().analysis).toBeNull(); expect(useStore.getState().loading).toBe(false);
});
