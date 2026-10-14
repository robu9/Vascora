import { test, expect } from '@playwright/test';
// Temporary: territory tuning screenshots. Delete before committing.
for (const sample of ['high-risk', 'low-risk']) {
  test(`tune ${sample}`, async ({ page }) => {
    await page.addInitScript(() => sessionStorage.setItem('vascora-disclaimer', '1'));
    await page.goto('/');
    await page.getByLabel('Sample patient').selectOption(sample);
    await expect(page.getByRole('button', { name: 'Risk overlay' })).toHaveAttribute('aria-pressed', 'true');
    await page.getByRole('button', { name: 'Pause beat' }).click();
    for (const view of ['Anterior', 'Posterior', 'Left', 'Right']) {
      await page.getByRole('button', { name: view, exact: true }).click();
      await page.waitForTimeout(1300);
      await page.locator('canvas').screenshot({ path: `test-results/tune-${sample}-${view.toLowerCase()}.png` });
    }
  });
}
