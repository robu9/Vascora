import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = path.dirname(fileURLToPath(import.meta.url));
const backend = path.resolve(frontend, '../backend');
const uvicorn = path.join(backend, process.platform === 'win32' ? '.venv/Scripts/uvicorn.exe' : '.venv/bin/uvicorn');

export default defineConfig({
  testDir: './e2e', timeout: 60000, workers: 1,
  use: { launchOptions: { timeout: 20000, args: process.platform === 'darwin' ? ['--use-angle=metal'] : [] }, baseURL: 'http://127.0.0.1:5173', viewport: { width: 1440, height: 900 }, screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: [
    { command: 'npm run dev -- --port 5173', cwd: frontend, url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI },
    { command: `"${uvicorn}" app.main:app --host 127.0.0.1 --port 8000`, cwd: backend, url: 'http://127.0.0.1:8000/api/health', timeout: 180000, reuseExistingServer: !process.env.CI },
  ],
});
