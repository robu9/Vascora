import { test, expect } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import { Mesh, PerspectiveCamera, Raycaster, Vector2, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
const ids = ['low-risk', 'moderate-risk', 'high-risk'];
test('live API, all patient presets, vessel selection, views and screenshot', async ({ page, request }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByLabel('Sample patient')).toBeEnabled();
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.getByText('Preparing anatomy…')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeDisabled();
  for (const id of ids) {
    const sample = await (await request.get(`http://127.0.0.1:8000/api/samples/${id}`)).json();
    const expected = await (await request.post('http://127.0.0.1:8000/api/analyze', { data: sample.features })).json();
    await page.getByLabel('Sample patient').selectOption(id);
    await expect(page.locator('.big-probability')).toContainText(`${(expected.prediction.cad.probability * 100).toFixed(1)}%`);
    for (const vessel of ['LAD', 'LCX', 'RCA']) {
      const button = page.locator('.vessel-card').filter({ hasText: vessel });
      await expect(button).toContainText(`${(expected.prediction.vessels[vessel].probability * 100).toFixed(1)}%`);
      await button.click(); await expect(button).toHaveAttribute('aria-pressed', 'true');
      await expect(page.locator('.detail-section')).toContainText(`${vessel} · DETAILS`);
    }
  }
  await expect(page.getByText('DEMO MODEL · HEURISTIC PLACEHOLDER')).toHaveCount(0);
  await expect(page.getByText('Confidence interval')).toBeVisible();
  await page.getByRole('button', { name: 'Pause beat' }).click();
  for (const view of ['Anterior', 'Posterior', 'Left', 'Right']) {
    await page.getByRole('button', { name: view, exact: true }).click();
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `test-results/${view.toLowerCase()}.png` });
  }
  await page.getByRole('button', { name: '↺ Reset' }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: 'test-results/viewer.png', fullPage: true });
  expect(errors).toEqual([]);
});
test('failed analysis stays neutral and can retry', async ({ page }) => {
  await page.route('**/api/analyze', route => route.fulfill({ status: 503, body: '{}' }));
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.getByRole('alert')).toContainText('503');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeDisabled();
  await page.unroute('**/api/analyze'); await page.getByRole('button', { name: 'Retry analysis' }).click();
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeEnabled();
});
test('missing GLB preserves numeric results', async ({ page }) => {
  await page.route('**/models/heart.glb', route => route.fulfill({ status: 404, body: '' }));
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.getByText('3D view unavailable')).toBeVisible();
  await expect(page.locator('.big-probability')).toContainText('%');
});
test('invalid mapping disables coloring, reduced motion and narrow layout', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/api/targets', async route => {
    const response = await route.fetch(); const body = await response.json(); body.targets.LAD.meshes.push('missing');
    await route.fulfill({ json: body });
  });
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.getByRole('alert')).toContainText('Risk overlay disabled');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Play beat' })).toBeDisabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});
test('zero probability is displayed and unavailable WebGL keeps results usable', async ({ page }) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type: string, ...args: unknown[]) {
      if (type.includes('webgl')) return null;
      return Reflect.apply(original, this, [type, ...args]);
    } as typeof original;
  });
  await page.route('**/api/analyze', async route => {
    const res = await route.fetch(); const body = await res.json();
    body.prediction.cad.probability = 0; body.prediction.cad.predicted_positive = false;
    body.prediction.vessels.LAD.probability = 0; body.prediction.vessels.LAD.predicted_positive = false;
    await route.fulfill({ json: body });
  });
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.locator('.big-probability')).toContainText('0.0%');
  await expect(page.getByText('3D view unavailable')).toBeVisible();
  await page.locator('.vessel-card').filter({ hasText: 'LAD' }).click();
  await expect(page.locator('.detail-section')).toContainText('0.0%');
});
test('rendering performance during orbit and heartbeat', async ({ page, browser }) => {
  await page.goto('/'); await expect(page.getByLabel('Sample patient')).toBeEnabled();
  await page.getByLabel('Sample patient').selectOption('moderate-risk');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeEnabled();
  await expect(page.getByText('Preparing anatomy…')).toHaveCount(0);
  await page.waitForTimeout(2500);
  const measured = page.evaluate(async () => {
    const canvas = document.querySelector('canvas')!;
    const gl = canvas.getContext('webgl2')!;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    const deltas: number[] = []; let last = performance.now(); const start = last;
    await new Promise<void>(resolve => {
      const frame = (now: number) => { deltas.push(now-last); last = now; if (now-start < 4000) requestAnimationFrame(frame); else resolve(); };
      requestAnimationFrame(frame);
    });
    return { renderer, fps: 1000 / (deltas.reduce((a,b) => a+b,0) / deltas.length), viewport: [innerWidth,innerHeight], frames: deltas.length, durationMs: last-start };
  });
  await page.mouse.move(500, 450); await page.mouse.down();
  await page.mouse.move(760, 480, { steps: 100 }); await page.mouse.move(450, 430, { steps: 100 }); await page.mouse.up();
  const report = { ...await measured, browser: browser.version(), platform: `${os.platform()} ${os.arch()}`, cpu: os.cpus()[0]?.model, headless: true, date: '2026-10-13' };
  await writeFile('../assets-pipeline/performance.json', JSON.stringify(report, null, 2) + '\n');
  expect(report.fps).toBeGreaterThanOrEqual(30);
});
test('a real artery mesh click and labels select the same target as vessel buttons', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/'); await expect(page.getByLabel('Sample patient')).toBeEnabled();
  await page.getByRole('button', { name: 'Pause beat' }).click();
  await expect(page.getByText('Preparing anatomy…')).toHaveCount(0);
  await page.getByRole('button', { name: 'Anterior', exact: true }).click();
  await page.evaluate(() => scrollTo(0, 0)); await page.waitForTimeout(1500);
  const rect = (await page.locator('canvas').boundingBox())!;
  const bytes = await readFile('public/models/heart.glb');
  const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
  const camera = new PerspectiveCamera(38, rect.width / rect.height, .1, 100);
  camera.position.set(0, .2, 4.5); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
  gltf.scene.updateMatrixWorld(true);
  const raycaster = new Raycaster(); let point: Vector3 | null = null;
  const mesh = gltf.scene.getObjectByName('LAD_mid') as Mesh;
  const position = mesh.geometry.getAttribute('position'); const indices = mesh.geometry.index!;
  for (let i = 0; i < indices.count && !point; i += 27) {
    const center = new Vector3();
    for (let j = 0; j < 3; j++) center.add(new Vector3().fromBufferAttribute(position, indices.getX(i+j)));
    center.divideScalar(3).applyMatrix4(mesh.matrixWorld).project(camera);
    raycaster.setFromCamera(new Vector2(center.x, center.y), camera);
    if (raycaster.intersectObject(gltf.scene, true)[0]?.object.name === 'LAD_mid') point = center;
  }
  expect(point).not.toBeNull();
  await page.mouse.click(rect.x + (point!.x + 1) * rect.width / 2, rect.y + (1 - point!.y) * rect.height / 2);
  await expect(page.locator('.vessel-card').filter({ hasText: 'LAD' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Labels', exact: true }).click();
  await expect(page.locator('.artery-label')).toHaveCount(3);
  await expect(page.locator('.artery-label').filter({ hasText: 'LAD' })).toBeVisible();
  await page.screenshot({ path: 'test-results/labels.png' });
  await page.getByRole('button', { name: 'Labels', exact: true }).click();
  await expect(page.locator('.artery-label')).toHaveCount(0);
  expect(errors).toEqual([]);
});
