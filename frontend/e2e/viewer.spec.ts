import { test, expect } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import os from 'node:os';
import { percent } from '../src/mapping';
const ids = ['low-risk', 'moderate-risk', 'high-risk'];
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => sessionStorage.setItem('vascora-disclaimer', '1'));
});
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
    await expect(page.locator('.big-probability')).toContainText(percent(expected.prediction.cad.probability));
    await page.waitForTimeout(800); // let vessel colors settle for the review screenshot
    await page.screenshot({ path: `test-results/preset-${id}.png` });
    for (const vessel of ['LAD', 'LCX', 'RCA']) {
      const button = page.locator('.vessel-card').filter({ hasText: vessel });
      await expect(button).toContainText(percent(expected.prediction.vessels[vessel].probability));
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
  await page.route('**/api/predict', route => route.fulfill({ status: 503, body: '{}' }));
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.getByRole('alert')).toContainText('503');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeDisabled();
  await page.unroute('**/api/predict'); await page.getByRole('button', { name: 'Retry analysis' }).click();
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
  await page.route('**/api/predict', async route => {
    const res = await route.fetch(); const body = await res.json();
    body.cad.probability = 0; body.cad.predicted_positive = false;
    body.vessels.LAD.probability = 0; body.vessels.LAD.predicted_positive = false;
    await route.fulfill({ json: body });
  });
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.locator('.big-probability')).toContainText('<1%');
  await expect(page.getByText('3D view unavailable')).toBeVisible();
  await page.locator('.vessel-card').filter({ hasText: 'LAD' }).click();
  await expect(page.locator('.detail-section')).toContainText('<1%');
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
  const report = { ...await measured, browser: browser.version(), platform: `${os.platform()} ${os.arch()}`, cpu: os.cpus()[0]?.model, headless: true, date: new Date().toISOString().slice(0, 10) };
  await writeFile('../assets-pipeline/performance.json', JSON.stringify(report, null, 2) + '\n');
  expect(report.fps).toBeGreaterThanOrEqual(30);
});
test('selecting an artery enables the risk tint on the realistic heart', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/'); await expect(page.getByLabel('Sample patient')).toBeEnabled();
  await page.getByRole('button', { name: 'Pause beat' }).click();
  await expect(page.getByText('Preparing anatomy…')).toHaveCount(0);
  await page.getByLabel('Sample patient').selectOption('low-risk');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeEnabled();
  await page.locator('.vessel-card').filter({ hasText: 'LAD' }).click();
  await expect(page.locator('.vessel-card').filter({ hasText: 'LAD' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Risk overlay' }).click();
  await expect(page.getByRole('button', { name: 'Labels', exact: true })).toHaveCount(0);
  await page.screenshot({ path: 'test-results/risk-heart.png' });
  expect(errors).toEqual([]);
});
test('clinical edits, explanations and held-out metrics stay tied to the heart', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Sample patient').selectOption('moderate-risk');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeEnabled();
  const before = await page.locator('.big-probability').innerText();
  await expect(page.getByRole('region', { name: 'Why this estimate' })).toContainText('Physiological measurements');
  await expect(page.getByRole('region', { name: 'Why this estimate' })).toContainText('Systolic Blood Pressure');
  await page.getByRole('button', { name: 'Show metrics' }).click();
  await expect(page.getByRole('table', { name: 'Classification metrics' })).toContainText('ROC-AUC');
  await expect(page.getByRole('table', { name: 'Classification metrics' })).toContainText('CAD');
  await page.getByLabel('Age', { exact: true }).fill('85');
  await expect(page.locator('.big-probability')).not.toHaveText(before, { timeout: 20000 });
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toHaveAttribute('aria-pressed', 'true');
});
test('empty or partial numbers are never scored and revert on blur', async ({ page }) => {
  const posted: Record<string, unknown>[] = [];
  page.on('request', r => { if (r.url().endsWith('/api/predict')) posted.push(r.postDataJSON()); });
  await page.goto('/'); await page.getByLabel('Sample patient').selectOption('moderate-risk');
  await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeEnabled();
  const age = page.getByLabel('Age', { exact: true });
  posted.length = 0;
  await age.fill('');
  await expect(age).toHaveValue('');
  await expect(page.getByText('Enter 18–100 years')).toBeVisible();
  await age.pressSequentially('67', { delay: 400 }); // "6" alone outlasts the 250 ms debounce
  await page.waitForTimeout(800);
  expect(posted.map(body => body.age)).toEqual([67]);
  await expect(page.locator('.intake [role=alert]')).toHaveCount(0);
  await age.fill('5'); await age.blur();
  await expect(age).toHaveValue('67');
});
test('the safety disclaimer is acknowledged before the first session', async ({ browser }) => {
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:5173/');
  const dialog = page.getByRole('dialog', { name: 'Decision support only' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(/not a substitute/i);
  await page.getByRole('button', { name: 'I understand' }).click();
  await expect(dialog).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.close();
});
for (const [width, height] of [[1440, 900], [1366, 768], [1024, 768], [390, 844]] as const) {
  test(`layout at ${width}x${height} keeps the heart and inputs readable`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.goto('/');
    await page.getByLabel('Sample patient').selectOption('high-risk');
    await expect(page.getByRole('button', { name: 'Risk overlay' })).toBeEnabled();
    await expect(page.getByText('Preparing anatomy…')).toHaveCount(0);
    await page.getByRole('button', { name: 'Pause beat' }).click();
    const box = async (selector: string) => (await page.locator(selector).first().boundingBox())!;
    const canvas = await box('canvas'), heading = await box('.viewer-heading'), controls = await box('.viewer-bottom');
    expect(heading.y + heading.height, 'heading ends above the heart').toBeLessThanOrEqual(canvas.y + 1);
    expect(controls.y, 'controls start below the heart').toBeGreaterThanOrEqual(canvas.y + canvas.height - 1);
    expect(canvas.height).toBeGreaterThan(300);
    if (width <= 1180) {
      expect((await box('#patient')).y, 'patient picker comes before the results').toBeLessThan((await box('.big-probability')).y);
    } else {
      expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight), 'page itself does not scroll').toBe(true);
      await page.locator('.results').evaluate(e => { e.scrollTop = e.scrollHeight; });
      expect((await box('canvas')).y, 'heart stays put while results scroll').toBe(canvas.y);
    }
    const tiny = await page.evaluate(() => [...document.querySelectorAll('body *')].filter(e => {
      const r = e.getBoundingClientRect();
      return r.width > 1 && r.height > 1 && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent!.trim()) && parseFloat(getComputedStyle(e).fontSize) < 11;
    }).map(e => `${e.tagName.toLowerCase()}.${e.className}`));
    expect(tiny, 'no visible text under 11px').toEqual([]);
    await page.screenshot({ path: `test-results/layout-${width}.png`, fullPage: width <= 1180 });
  });
}
test('one finger on the heart scrolls the page on touch screens', async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  await context.addInitScript(() => sessionStorage.setItem('vascora-disclaimer', '1'));
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:5173/');
  await expect(page.getByText('Two fingers to rotate')).toBeVisible();
  await expect(page.getByText('Preparing anatomy…')).toHaveCount(0);
  const chain = await page.locator('canvas').evaluate(c => {
    const out: string[] = [];
    for (let e: Element | null = c; e && !e.classList.contains('stage'); e = e.parentElement) out.push(getComputedStyle(e).touchAction);
    return out;
  });
  expect(chain, 'nothing between the canvas and the page blocks touch panning').not.toContain('none');
  const canvas = (await page.locator('canvas').boundingBox())!;
  const cdp = await context.newCDPSession(page);
  await cdp.send('Input.synthesizeScrollGesture', { x: canvas.x + canvas.width / 2, y: canvas.y + canvas.height * 0.7, yDistance: -300, gestureSourceType: 'touch', speed: 1200 });
  await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(100);
  await context.close();
});
