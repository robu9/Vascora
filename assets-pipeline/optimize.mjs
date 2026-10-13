import { NodeIO } from '../frontend/node_modules/@gltf-transform/core/dist/index.js';
import { ALL_EXTENSIONS } from '../frontend/node_modules/@gltf-transform/extensions/dist/index.js';
import { meshopt } from '../frontend/node_modules/@gltf-transform/functions/dist/index.js';
import { MeshoptEncoder, MeshoptDecoder } from '../frontend/node_modules/meshoptimizer/index.js';
import { fileURLToPath } from 'node:url';
await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(fileURLToPath(new URL('./heart.raw.glb', import.meta.url)));
// Avoid join/flatten/dedup-materials: node identities and independent vessel materials matter.
await doc.transform(meshopt({ encoder: MeshoptEncoder, level: 'medium' }));
await io.write(fileURLToPath(new URL('../frontend/public/models/heart.glb', import.meta.url)), doc);
console.log('Wrote optimized heart.glb');
