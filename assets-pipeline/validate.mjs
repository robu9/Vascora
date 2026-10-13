import { NodeIO, getBounds } from '../frontend/node_modules/@gltf-transform/core/dist/index.js';
import { ALL_EXTENSIONS } from '../frontend/node_modules/@gltf-transform/extensions/dist/index.js';
import { MeshoptDecoder } from '../frontend/node_modules/meshoptimizer/index.js';
import { readFile, stat, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
const url = new URL('../frontend/public/models/heart.glb', import.meta.url);
const registry = JSON.parse(await readFile(new URL('../shared/vessel_registry.json', import.meta.url), 'utf8'));
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(fileURLToPath(url));
const nodes = doc.getRoot().listNodes().filter(n => n.getMesh());
const materials = new Set();
const tissue = nodes.filter(n => /ventricle|atrium/.test(n.getName()));
const tissueBounds = tissue.map(getBounds);
const min = [0, 1, 2].map(i => Math.min(...tissueBounds.map(b => b.min[i])));
const max = [0, 1, 2].map(i => Math.max(...tissueBounds.map(b => b.max[i])));
assert.ok(max[1] - min[1] > 1, 'Heart surfaces must occupy a substantial part of the normalized model');
for (const target of Object.values(registry.targets)) for (const name of target.meshes ?? []) {
  const matching = nodes.filter(n => n.getName() === name);
  assert.equal(matching.length, 1, `Exactly one node required: ${name}`);
  const bounds = getBounds(matching[0]);
  for (let axis = 0; axis < 3; axis++) {
    assert.ok(bounds.min[axis] >= min[axis] - .15 && bounds.max[axis] <= max[axis] + .15, `${name} must remain aligned with heart surfaces`);
  }
  for (const primitive of matching[0].getMesh().listPrimitives()) {
    const mat = primitive.getMaterial(); assert.ok(mat, `${name} has a material`);
    assert.ok(!materials.has(mat), `${name} has an independent material`); materials.add(mat);
  }
}
let triangles = 0;
for (const mesh of doc.getRoot().listMeshes()) for (const primitive of mesh.listPrimitives()) {
  assert.equal(primitive.getMode(), 4, 'Triangle geometry required');
  triangles += (primitive.getIndices()?.getCount() ?? primitive.getAttribute('POSITION').getCount()) / 3;
}
const bytes = (await stat(url)).size;
assert.ok(triangles < 150000, `Triangle budget: ${triangles}`);
assert.ok(bytes < 8 * 1024 * 1024, `Payload budget: ${bytes}`);
assert.ok(doc.getRoot().listExtensionsUsed().some(e => e.extensionName === 'EXT_meshopt_compression'));
const report = { meshCount: nodes.length, requiredMeshes: materials.size, triangles, bytes, meshopt: true, textures: doc.getRoot().listTextures().length };
await writeFile(new URL('./validation.json', import.meta.url), JSON.stringify(report, null, 2) + '\n');
console.log(report);
