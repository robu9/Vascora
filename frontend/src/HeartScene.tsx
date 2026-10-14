import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { CameraControls, CameraControlsImpl, PerformanceMonitor, useGLTF, useProgress } from '@react-three/drei';
import { Box3, Color, Group, Mesh, MeshStandardMaterial, PMREMGenerator, Vector3 } from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { useStore } from './store';
import { riskColor, validateMapping } from './mapping';
import type { Vessel } from './types';

type View = 'Anterior' | 'Posterior' | 'Left' | 'Right';
const positions: Record<View, [number, number, number]> = {
  Anterior: [0, 0.2, 4.5], Posterior: [0, 0.2, -4.5], Left: [4.5, 0.2, 0], Right: [-4.5, 0.2, 0],
};
const tissueNames = new Set(['Left_ventricle', 'Right_ventricle', 'Left_atrium', 'Right_atrium']);
const neutral = new Color('#ffffff');
// Touch screens: one finger scrolls the page, two fingers rotate and zoom the heart.
const coarsePointer = matchMedia('(pointer: coarse)').matches;
const { ACTION } = CameraControlsImpl;
const touchControls = coarsePointer ? { touches: { one: ACTION.NONE, two: ACTION.TOUCH_DOLLY_ROTATE, three: ACTION.NONE } } : {};
function StudioEnvironment() {
  const gl = useThree(s => s.gl);
  const scene = useThree(s => s.scene);
  useEffect(() => {
    const pmrem = new PMREMGenerator(gl);
    const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = environment;
    return () => { scene.environment = null; environment.dispose(); pmrem.dispose(); };
  }, [gl, scene]);
  return null;
}
function LoadingOverlay() {
  const active = useProgress(s => s.active);
  return active ? <div className="model-progress" role="status">Preparing anatomy…</div> : null;
}
class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <div className="scene-fallback" role="alert"><h2>3D view unavailable</h2><p>The model could not load or this browser cannot render WebGL.</p><p>Patient results remain available.</p><button onClick={() => location.reload()}>Reload viewer</button></div> : this.props.children; }
}
function Heart({ playing, bpm, view, reset, onError }: {
  playing: boolean; bpm: number; view: View; reset: number; onError: (s: string | null) => void;
}) {
  const { scene: arterySource } = useGLTF('/models/heart.glb');
  const { scene: shellSource } = useGLTF('/models/realistic_human_heart.glb');
  const scene = useMemo(() => {
    const clone = arterySource.clone(true);
    clone.traverse(o => { if (o instanceof Mesh) o.visible = false; });
    return clone;
  }, [arterySource]);
  const shell = useMemo(() => {
    const clone = shellSource.clone(true);
    clone.traverse(o => {
      if (!(o instanceof Mesh)) return;
      const sources = Array.isArray(o.material) ? o.material : [o.material];
      const cloned = sources.map(material => {
        const copy = material.clone();
        if (copy instanceof MeshStandardMaterial) copy.userData.baseColor = copy.color.clone();
        return copy;
      });
      o.material = Array.isArray(o.material) ? cloned : cloned[0];
    });
    return clone;
  }, [shellSource]);
  useLayoutEffect(() => {
    scene.updateMatrixWorld(true);
    const tissue = new Box3();
    scene.traverse(o => { if (o instanceof Mesh && tissueNames.has(o.name)) tissue.union(new Box3().setFromObject(o)); });
    shell.position.set(0, 0, 0); shell.scale.set(1, 1, 1); shell.updateMatrixWorld(true);
    const current = new Box3().setFromObject(shell);
    const scale = tissue.getSize(new Vector3()).y / current.getSize(new Vector3()).y;
    shell.scale.setScalar(scale); shell.updateMatrixWorld(true);
    const fitted = new Box3().setFromObject(shell);
    shell.position.copy(tissue.getCenter(new Vector3()).sub(fitted.getCenter(new Vector3())));
  }, [scene, shell]);
  useEffect(() => () => shell.traverse(o => {
    if (!(o instanceof Mesh)) return;
    const materials = Array.isArray(o.material) ? o.material : [o.material];
    materials.forEach(material => material.dispose());
  }), [shell]);
  const registry = useStore(s => s.registry), selected = useStore(s => s.selected);
  const analysis = useStore(s => s.analysis), mode = useStore(s => s.mode);
  const mapping = useMemo(() => {
    const names: string[] = [];
    scene.traverse(o => { if (o instanceof Mesh) names.push(o.name); });
    return registry ? validateMapping(registry, names) : { lookup: new Map<string, Vessel>(), error: null };
  }, [scene, registry]);
  useEffect(() => { onError(mapping.error); }, [mapping.error, onError]);
  const group = useRef<Group>(null), controls = useRef<CameraControls>(null);
  const invalidate = useThree(s => s.invalidate);
  const connected = useThree(s => s.events.connected as HTMLElement | undefined), gl = useThree(s => s.gl);
  useEffect(() => {
    // camera-controls sets touch-action:none on the element it listens to; let vertical pans scroll the page.
    if (coarsePointer) for (const element of [connected, gl.domElement]) if (element) element.style.touchAction = 'pan-y';
  }, [connected, gl]);
  const phase = useRef(0), transition = useRef(0);
  useEffect(() => { transition.current = 1; invalidate(); }, [mode, analysis, selected, mapping, invalidate]);
  useEffect(() => {
    const [x, y, z] = positions[view];
    void controls.current?.setLookAt(x, y, z, 0, 0, 0, true);
  }, [view, reset]);
  useFrame((_, delta) => {
    if (group.current) {
      if (playing) phase.current += Math.min(delta, 0.1) * bpm / 60 * Math.PI * 2;
      const beat = playing ? Math.pow(Math.max(0, Math.sin(phase.current)), 4) * 0.018 : 0;
      group.current.scale.set(1 - beat, 1 + beat * 0.45, 1 - beat);
    }
    const probability = selected ? analysis?.prediction.vessels[selected]?.probability : undefined;
    const risk = mode === 'risk' && !mapping.error && probability != null;
    const tint = risk ? new Color(riskColor(probability)) : neutral;
    const ease = 1 - Math.exp(-delta * 9);
    shell.traverse(o => {
      if (!(o instanceof Mesh)) return;
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const material of materials) {
        if (!(material instanceof MeshStandardMaterial)) continue;
        const base = material.userData.baseColor as Color | undefined;
        if (!base) continue;
        material.color.lerp(risk ? base.clone().lerp(tint, 0.55) : base, ease);
        material.emissive.copy(tint);
        material.emissiveIntensity = risk ? 0.22 : 0;
      }
    });
    if (transition.current > 0) { transition.current -= delta; invalidate(); }
  });
  return <>
    <CameraControls ref={controls} minDistance={1.7} maxDistance={7} smoothTime={0.3} {...touchControls}/>
    <group ref={group}>
      <primitive object={shell}/>
      <primitive object={scene}/>
    </group>
  </>;
}
export function HeartScene() {
  const [view, setView] = useState<View>('Anterior'), [reset, setReset] = useState(0);
  const [paused, setPaused] = useState(false);
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
  const riskUnavailable = assetError ? 'Unavailable: the 3D model does not match the vessel registry' : !analysis ? 'Available once an estimate is ready' : undefined;
  return <section className="viewer" aria-label="Interactive anatomical heart">
    <div className="viewer-heading">
      <div><span className="eyebrow">ANATOMICAL EXPLORER <span className="organ-index">· 01 / HEART</span></span><h1>A closer look at the heart.</h1></div>
      <div className="mode-control"><button aria-pressed={mode === 'anatomy'} onClick={() => setMode('anatomy')}>Anatomy</button><button aria-pressed={mode === 'risk'} disabled={!!riskUnavailable} title={riskUnavailable} onClick={() => setMode('risk')}>Risk overlay</button></div>
    </div>
    <div className="stage">
      <SceneBoundary><Canvas camera={{ position: positions.Anterior, fov: 38 }} dpr={dpr} frameloop={playing ? 'always' : 'demand'} gl={{ antialias: true, alpha: true }} fallback={<div className="scene-fallback">WebGL is unavailable. Patient results are shown alongside.</div>}>
        <StudioEnvironment/>
        <ambientLight intensity={0.22}/><hemisphereLight args={['#fff1e6', '#1a3030', 0.4]}/>
        <directionalLight position={[-3, 4, 5]} intensity={2.4} color="#fff5ee"/>
        <directionalLight position={[3.5, 1.2, -3]} intensity={0.85} color="#c5e4e4"/>
        <PerformanceMonitor onDecline={() => setDpr(1)} onIncline={() => setDpr(1.5)}/>
        <Suspense fallback={null}><Heart playing={playing} bpm={bpm} view={view} reset={reset} onError={setAssetError}/></Suspense>
      </Canvas></SceneBoundary>
      <LoadingOverlay/>
      {assetError && <p className="asset-error" role="alert">Risk overlay disabled: {assetError}</p>}
    </div>
    <div className="viewer-bottom"><div className="view-presets">{(Object.keys(positions) as View[]).map(v => <button key={v} aria-pressed={view === v} onClick={() => { setView(v); setReset(n => n + 1); }}>{v}</button>)}</div><div className="view-tools"><button onClick={() => { setView('Anterior'); setReset(n => n + 1); useStore.getState().selectVessel(null); }}>↺ Reset</button><button aria-pressed={!paused && !reduced} disabled={reduced} title={reduced ? 'Paused by your reduced-motion setting' : undefined} onClick={() => setPaused(v => !v)}>{paused || reduced ? 'Play beat' : 'Pause beat'}</button></div></div>
    <div className="canvas-caption"><span>{coarsePointer ? 'Two fingers to rotate · pinch to zoom' : 'Drag to rotate · scroll to zoom'}</span><span>{playing ? `${bpm} BPM · illustrative motion` : 'Motion paused'}</span></div>
  </section>;
}
