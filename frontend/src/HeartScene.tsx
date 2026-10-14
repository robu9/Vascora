import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { CameraControls, Html, PerformanceMonitor, useGLTF, useProgress } from '@react-three/drei';
import { Box3, Color, Group, Mesh, MeshPhysicalMaterial, Vector3, type WebGLProgramParametersWithUniforms } from 'three';
import { useStore } from './store';
import { validateMapping } from './mapping';
import type { Vessel } from './types';

type View = 'Anterior' | 'Posterior' | 'Left' | 'Right';
const positions: Record<View, [number, number, number]> = {
  Anterior: [0, 0.2, 4.5], Posterior: [0, 0.2, -4.5], Left: [4.5, 0.2, 0], Right: [-4.5, 0.2, 0],
};
const normal = new Color('#d7a18e');
const low = new Color('#4477aa'), high = new Color('#ee9955');
function LoadingOverlay() {
  const active = useProgress(s => s.active);
  return active ? <div className="model-progress" role="status">Preparing anatomy…</div> : null;
}
class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <div className="scene-fallback" role="alert"><h2>3D view unavailable</h2><p>The model could not load or this browser cannot render WebGL.</p><p>Patient results remain available.</p><button onClick={() => location.reload()}>Reload viewer</button></div> : this.props.children; }
}
function Heart({ playing, bpm, labels, view, reset, onError }: {
  playing: boolean; bpm: number; labels: boolean; view: View; reset: number; onError: (s: string | null) => void;
}) {
  const { scene: source } = useGLTF('/models/heart.glb');
  const scene = useMemo(() => {
    const clone = source.clone(true);
    clone.traverse(o => {
      if (o instanceof Mesh) {
        const original = Array.isArray(o.material) ? o.material[0] : o.material;
        o.material = new MeshPhysicalMaterial({ color: original.color ?? '#803a32', roughness: 0.48, metalness: 0, clearcoat: 0.18, clearcoatRoughness: 0.5 });
        if (['Left_ventricle', 'Right_ventricle', 'Left_atrium', 'Right_atrium'].includes(o.name)) {
          o.material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
            shader.vertexShader = 'varying vec3 vSurface;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvSurface = position;');
            shader.fragmentShader = 'varying vec3 vSurface;\n' + shader.fragmentShader;
            shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
              float grain = sin(vSurface.x * 160.0) * sin(vSurface.y * 137.0) * sin(vSurface.z * 151.0);
              float mottling = sin(vSurface.x * 17.0 + sin(vSurface.y * 21.0)) * sin(vSurface.z * 19.0);
              diffuseColor.rgb *= 0.96 + 0.025 * grain + 0.05 * mottling;`);
          };
          o.material.customProgramCacheKey = () => 'vascora-tissue-v1';
        }
      }
    });
    return clone;
  }, [source]);
  useEffect(() => () => scene.traverse(o => { if (o instanceof Mesh) (o.material as MeshPhysicalMaterial).dispose(); }), [scene]);
  const registry = useStore(s => s.registry), selected = useStore(s => s.selected);
  const analysis = useStore(s => s.analysis), mode = useStore(s => s.mode);
  const select = useStore(s => s.selectVessel);
  const mapping = useMemo(() => {
    const names: string[] = [];
    scene.traverse(o => { if (o instanceof Mesh) names.push(o.name); });
    return registry ? validateMapping(registry, names) : { lookup: new Map<string, Vessel>(), error: null };
  }, [scene, registry]);
  useEffect(() => { onError(mapping.error); }, [mapping.error, onError]);
  const group = useRef<Group>(null), controls = useRef<CameraControls>(null);
  const [hovered, setHovered] = useState<Vessel | null>(null);
  const invalidate = useThree(s => s.invalidate);
  const phase = useRef(0), transition = useRef(0);
  useEffect(() => { transition.current = 1; invalidate(); }, [mode, analysis, selected, hovered, mapping, invalidate]);
  useEffect(() => {
    const [x, y, z] = positions[view];
    void controls.current?.setLookAt(x, y, z, 0, 0, 0, true);
  }, [view, reset]);
  const boxes = useMemo(() => {
    const result = new Map<Vessel, Box3>();
    scene.updateMatrixWorld(true);
    scene.traverse(o => {
      const v = mapping.lookup.get(o.name);
      if (v && o instanceof Mesh) {
        const box = result.get(v) ?? new Box3(); box.union(new Box3().setFromObject(o)); result.set(v, box);
      }
    });
    return result;
  }, [scene, mapping]);
  useEffect(() => {
    if (!selected) return;
    const box = boxes.get(selected);
    if (!box) return;
    const center = box.getCenter(new Vector3());
    const direction = center.clone().setY(0).normalize();
    if (direction.length() < 0.1) direction.set(0, 0, 1);
    const eye = center.clone().addScaledVector(direction, 3.3); eye.y += 0.4;
    void controls.current?.setLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, true);
  }, [selected, boxes]);
  useFrame((_, delta) => {
    if (group.current) {
      if (playing) phase.current += Math.min(delta, 0.1) * bpm / 60 * Math.PI * 2;
      const beat = playing ? Math.pow(Math.max(0, Math.sin(phase.current)), 4) * 0.018 : 0;
      group.current.scale.set(1 - beat, 1 + beat * 0.45, 1 - beat);
    }
    scene.traverse(o => {
      if (!(o instanceof Mesh)) return;
      const vessel = mapping.lookup.get(o.name);
      if (!vessel) return;
      const material = o.material as MeshPhysicalMaterial;
      const probability = analysis?.prediction.vessels[vessel]?.probability;
      const color = mode === 'risk' && !mapping.error && probability != null ? low.clone().lerp(high, probability) : normal;
      material.color.lerp(color, 1 - Math.exp(-delta * 9));
      material.emissive.copy(color);
      material.emissiveIntensity = vessel === selected ? 0.32 : vessel === hovered ? 0.18 : 0;
    });
    if (transition.current > 0) { transition.current -= delta; invalidate(); }
  });
  const vesselEvent = (event: ThreeEvent<PointerEvent | MouseEvent>) => {
    const vessel = mapping.lookup.get(event.object.name);
    if (vessel) event.stopPropagation();
    return vessel;
  };
  return <>
    <CameraControls ref={controls} minDistance={1.7} maxDistance={7} smoothTime={0.3}/>
    <group ref={group}>
      <primitive object={scene}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => setHovered(vesselEvent(e) ?? null)}
        onPointerOut={() => setHovered(null)}
        onClick={(e: ThreeEvent<MouseEvent>) => { const vessel = vesselEvent(e); if (vessel) select(vessel); }}/>
      {labels && [...boxes].map(([vessel, box]) => <Html key={vessel} position={box.getCenter(new Vector3())} center zIndexRange={[5, 0]} distanceFactor={4}><button className={`artery-label ${selected === vessel ? 'active' : ''}`} onClick={() => select(vessel)}>{vessel}</button></Html>)}
    </group>
  </>;
}
export function HeartScene() {
  const [view, setView] = useState<View>('Anterior'), [reset, setReset] = useState(0);
  const [labels, setLabels] = useState(false), [paused, setPaused] = useState(false);
  const [hidden, setHidden] = useState(document.hidden), [reduced, setReduced] = useState(matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [dpr, setDpr] = useState(1.5), [assetError, setAssetError] = useState<string | null>(null);
  const mode = useStore(s => s.mode), setMode = useStore(s => s.setMode), analysis = useStore(s => s.analysis);
  // A primitive selector: other form edits do not re-render the canvas parent.
  const bpm = useStore(s => Number(s.features.pr ?? s.samples.find(x => x.id === s.sampleId)?.features.pr ?? s.schema?.features.find(f => f.key === 'pr')?.default ?? 75));
  useEffect(() => {
    const visibility = () => setHidden(document.hidden);
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const motion = () => setReduced(media.matches);
    document.addEventListener('visibilitychange', visibility); media.addEventListener('change', motion);
    return () => { document.removeEventListener('visibilitychange', visibility); media.removeEventListener('change', motion); };
  }, []);
  const playing = !paused && !hidden && !reduced;
  return <section className="viewer" aria-label="Interactive anatomical heart">
    <div className="viewer-heading"><div><span className="eyebrow">ANATOMICAL EXPLORER</span><h1>A closer look<br/>at the heart.</h1><p>Explore the coronary arteries and their estimated risk.</p></div><span className="organ-index">01 / HEART</span></div>
    <div className="mode-control"><button aria-pressed={mode === 'anatomy'} onClick={() => setMode('anatomy')}>Anatomy</button><button aria-pressed={mode === 'risk'} disabled={!analysis || !!assetError} onClick={() => setMode('risk')}>Risk overlay</button></div>
    <SceneBoundary><Canvas camera={{ position: positions.Anterior, fov: 38 }} dpr={dpr} frameloop={playing ? 'always' : 'demand'} gl={{ antialias: true, alpha: true }} fallback={<div className="scene-fallback">WebGL is unavailable. Patient results are shown alongside.</div>}>
      <ambientLight intensity={0.8}/><hemisphereLight args={['#ffe6d4', '#1a3737', 1.8]}/>
      <directionalLight position={[-3, 4, 5]} intensity={3.2} color="#ffe1cf"/>
      <directionalLight position={[4, 1, -3]} intensity={2.8} color="#a5d3d3"/>
      <PerformanceMonitor onDecline={() => setDpr(1)} onIncline={() => setDpr(1.5)}/>
      <Suspense fallback={null}><Heart playing={playing} bpm={bpm} labels={labels} view={view} reset={reset} onError={setAssetError}/></Suspense>
    </Canvas></SceneBoundary>
    <LoadingOverlay/>
    {assetError && <p className="asset-error" role="alert">Risk overlay disabled: {assetError}</p>}
    <div className="viewer-bottom"><div className="view-presets">{(Object.keys(positions) as View[]).map(v => <button key={v} aria-pressed={view === v} onClick={() => { setView(v); setReset(n => n + 1); }}>{v}</button>)}</div><div className="view-tools"><button onClick={() => { setView('Anterior'); setReset(n => n + 1); useStore.getState().selectVessel(null); }}>↺ Reset</button><button aria-pressed={labels} onClick={() => setLabels(v => !v)}>Labels</button><button aria-pressed={!paused && !reduced} disabled={reduced} onClick={() => setPaused(v => !v)}>{paused || reduced ? 'Play beat' : 'Pause beat'}</button></div></div>
    <div className="canvas-caption"><span>DRAG TO ROTATE · SCROLL TO ZOOM</span><span>{playing ? `${bpm} BPM · Illustrative motion` : 'Motion paused'}</span></div>
  </section>;
}
