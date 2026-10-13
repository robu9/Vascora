import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useStore } from './store';
import type { Analysis } from './types';
const sample = (id: string) => ({ id, name: id, description: '', features: { age: 60 } });
const answer = (name: string) => ({ prediction: { model: { name } }, explanations: {} }) as Analysis;
beforeEach(() => useStore.setState({ samples: [sample('a'), sample('b')], analysis: null, loading: false, error: null, mode: 'anatomy' }));
afterEach(() => vi.unstubAllGlobals());
it('clears old results immediately when changing patients', async () => {
  useStore.setState({ analysis: answer('old') });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(answer('new')))));
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
  resolvers[1](new Response(JSON.stringify(answer('b')))); await second;
  resolvers[0](new Response(JSON.stringify(answer('a')))); await first;
  expect(useStore.getState().analysis?.prediction.model.name).toBe('b');
});
it('leaves neutral anatomy available after failure and supports retry', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('', { status: 503 })).mockResolvedValueOnce(new Response(JSON.stringify(answer('retry')))));
  await useStore.getState().selectSample('a');
  expect(useStore.getState().error).toContain('503'); expect(useStore.getState().analysis).toBeNull();
  await useStore.getState().selectSample('a'); expect(useStore.getState().error).toBeNull();
});
it('returning to anatomy cancels pending analysis', async () => {
  let resolve!: (r: Response) => void;
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(r => { resolve = r; })));
  const first = useStore.getState().selectSample('a');
  await useStore.getState().selectSample(''); resolve(new Response(JSON.stringify(answer('a')))); await first;
  expect(useStore.getState().analysis).toBeNull(); expect(useStore.getState().loading).toBe(false);
});
