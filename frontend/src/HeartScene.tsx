import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { CameraControls, CameraControlsImpl, Html, PerformanceMonitor, useGLTF, useProgress } from '@react-three/drei';
import { Box3, Color, Group, Mesh, MeshStandardMaterial, PMREMGenerator, Vector2, Vector3, type WebGLProgramParametersWithUniforms } from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { useStore } from './store';
import { riskColor, validateMapping, vessels } from './mapping';
import type { Vessel } from './types';

type View = 'Anterior' | 'Posterior' | 'Left' | 'Right';
const positions: Record<View, [number, number, number]> = {
  Anterior: [0, 0.2, 4.5], Posterior: [0, 0.2, -4.5], Left: [4.5, 0.2, 0], Right: [-4.5, 0.2, 0],
};
const neutral = new Color('#d7a18e');
const tissueNames = new Set(['Left_ventricle', 'Right_ventricle', 'Left_atrium', 'Right_atrium']);
// Model axes (assets-pipeline/README.md): +X patient left, +Y superior, +Z anterior. Each artery's registry
// territory is the wall it typically supplies; these are the directions those walls face from the ventricles.
const territoryAxes: Record<string, Vector3> = {
  anterior: new Vector3(0.3, -0.2, 1).normalize(),
  lateral: new Vector3(1, 0.05, -0.45).normalize(),
  inferior: new Vector3(-0.7, -0.6, -0.25).normalize(),
};
// Labels sit at each territory's outermost point toward the side its artery runs on. The LCX wall wraps to the
// back, so its side is turned slightly forward to keep all three labels on the front of the heart.
const labelSide: Record<Vessel, Vector3> = {
  LAD: new Vector3(0, 0, 1), LCX: new Vector3(1, 0, 0.3).normalize(), RCA: new Vector3(-1, 0, 0),
};
/** Index of the territory a direction from the ventricles falls in. */
function territoryIndex(direction: Vector3, axes: Vector3[]) {
  let best = -1, bestScore = 0;
  axes.forEach((axis, i) => {
    const score = axis ? direction.dot(axis) : 0;
    if (score > bestScore) { bestScore = score; best = i; }
  });
  return best;
}
// Touch screens: one finger scrolls the page, two fingers rotate and zoom the heart.
const coarsePointer = matchMedia('(pointer: coarse)').matches;
const { ACTION } = CameraControlsImpl;
const touchControls = coarsePointer ? { touches: { one: ACTION.NONE, two: ACTION.TOUCH_DOLLY_ROTATE, three: ACTION.NONE } } : {};

type RiskUniforms = ReturnType<typeof createRiskUniforms>;
function createRiskUniforms() {
  return {
    uRiskCenter: { value: new Vector3() }, uRiskBand: { value: new Vector2() }, uRiskStrength: { value: 0 },
    uRiskAxes: { value: vessels.map(() => new Vector3(0, 0, 1)) },
    uRiskColors: { value: vessels.map(() => neutral.clone()) },
    uRiskFocus: { value: new Vector3() },
  };
}
/**
 * Tints the textured heart toward each artery's color over the wall that artery supplies, keeping the texture's
 * light and shade. Above the band (atria and great vessels) the surface keeps its own color.
 */
function territoryShader(uniforms: RiskUniforms) {
  return (shader: WebGLProgramParametersWithUniforms) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = 'varying vec3 vRiskWorld;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvRiskWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = `varying vec3 vRiskWorld;
uniform vec3 uRiskCenter; uniform vec2 uRiskBand; uniform float uRiskStrength;
uniform vec3 uRiskAxes[3]; uniform vec3 uRiskColors[3]; uniform vec3 uRiskFocus;
` + shader.fragmentShader
      .replace('#include <color_fragment>', `#include <color_fragment>
        vec3 riskGlow = vec3(0.0);
        {
          vec3 d = normalize(vRiskWorld - uRiskCenter);
          vec3 w = pow(max(vec3(dot(d, uRiskAxes[0]), dot(d, uRiskAxes[1]), dot(d, uRiskAxes[2])), 0.0), vec3(3.0));
          float total = w.x + w.y + w.z;
          if (total > 1e-4) {
            vec3 tint = (uRiskColors[0] * w.x + uRiskColors[1] * w.y + uRiskColors[2] * w.z) / total;
            float focus = dot(w, uRiskFocus) / total;
            float k = uRiskStrength * (1.0 - smoothstep(uRiskBand.x, uRiskBand.y, vRiskWorld.y))
              * smoothstep(0.02, 0.25, max(w.x, max(w.y, w.z))) * (0.5 + 0.25 * focus);
            // Shade the tint by the texture's own brightness so vessels, fat and folds stay visible through it.
            float lum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));
            diffuseColor.rgb = mix(diffuseColor.rgb, tint * clamp(0.2 + 2.4 * lum, 0.0, 1.6), k);
            riskGlow = tint * k * 0.18 * focus;
          }
        }`)
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += riskGlow;');
  };
}
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
function Heart({ playing, bpm, labels, view, reset, onError }: {
  playing: boolean; bpm: number; labels: boolean; view: View; reset: number; onError: (s: string | null) => void;
}) {
  // The segmented Z-Anatomy model is never drawn: it holds the vessel-name contract and sets the heart's size.
  const { scene: contractSource } = useGLTF('/models/heart.glb');
  const { scene: shellSource } = useGLTF('/models/realistic_human_heart.glb');
  const uniforms = useMemo(createRiskUniforms, []);
  const { shell, frame } = useMemo(() => {
    const shell = shellSource.clone(true);
    shell.traverse(o => {
      if (!(o instanceof Mesh)) return;
      const cloned = (Array.isArray(o.material) ? o.material : [o.material]).map(material => {
        const copy = material.clone();
        if (copy instanceof MeshStandardMaterial) {
          copy.onBeforeCompile = territoryShader(uniforms);
          copy.customProgramCacheKey = () => 'vascora-territory-v1';
        }
        return copy;
      });
      o.material = Array.isArray(o.material) ? cloned : cloned[0];
    });
    // Size and center the realistic heart on the segmented model's chambers, which the camera presets frame.
    contractSource.updateMatrixWorld(true);
    const tissue = new Box3();
    contractSource.traverse(o => { if (o instanceof Mesh && tissueNames.has(o.name)) tissue.union(new Box3().setFromObject(o)); });
    shell.updateMatrixWorld(true);
    shell.scale.setScalar(tissue.getSize(new Vector3()).y / new Box3().setFromObject(shell).getSize(new Vector3()).y);
    shell.updateMatrixWorld(true);
    shell.position.copy(tissue.getCenter(new Vector3()).sub(new Box3().setFromObject(shell).getCenter(new Vector3())));
    shell.updateMatrixWorld(true);
    // Territory directions are measured from the ventricles; the band above them fades out the tint.
    const box = new Box3().setFromObject(shell), size = box.getSize(new Vector3()), mid = box.getCenter(new Vector3());
    const frame = {
      center: new Vector3(mid.x, box.min.y + 0.38 * size.y, mid.z),
      band: new Vector2(box.min.y + 0.52 * size.y, box.min.y + 0.7 * size.y),
    };
    return { shell, frame };
  }, [contractSource, shellSource, uniforms]);
  useEffect(() => () => shell.traverse(o => {
    if (o instanceof Mesh) (Array.isArray(o.material) ? o.material : [o.material]).forEach(material => material.dispose());
  }), [shell]);
  const registry = useStore(s => s.registry), selected = useStore(s => s.selected);
  const analysis = useStore(s => s.analysis), mode = useStore(s => s.mode);
  const select = useStore(s => s.selectVessel);
  const mapping = useMemo(() => {
    const names: string[] = [];
    contractSource.traverse(o => { if (o instanceof Mesh) names.push(o.name); });
    if (!registry) return { lookup: new Map<string, Vessel>(), axes: [] as Vector3[], error: null };
    const { lookup, error } = validateMapping(registry, names);
    const axes = vessels.map(v => territoryAxes[registry.targets[v]?.territory ?? '']);
    const missing = vessels.filter((_, i) => !axes[i]);
    return { lookup, axes, error: error ?? (missing.length ? `No heart territory for ${missing.join(', ')}` : null) };
  }, [contractSource, registry]);
  useEffect(() => { onError(mapping.error); }, [mapping.error, onError]);
  useEffect(() => {
    uniforms.uRiskCenter.value.copy(frame.center);
    uniforms.uRiskBand.value.copy(frame.band);
    mapping.axes.forEach((axis, i) => { if (axis) uniforms.uRiskAxes.value[i].copy(axis); });
  }, [uniforms, frame, mapping]);
  const group = useRef<Group>(null), controls = useRef<CameraControls>(null);
  const [hovered, setHovered] = useState<Vessel | null>(null);
  const invalidate = useThree(s => s.invalidate);
  const connected = useThree(s => s.events.connected as HTMLElement | undefined), gl = useThree(s => s.gl);
  useEffect(() => {
    // camera-controls sets touch-action:none on the element it listens to; let vertical pans scroll the page.
    if (coarsePointer) for (const element of [connected, gl.domElement]) if (element) element.style.touchAction = 'pan-y';
  }, [connected, gl]);
  const phase = useRef(0), transition = useRef(0);
  const risk = mode === 'risk' && !mapping.error && !!analysis;
  const colors = useMemo(() => vessels.map(v => {
    const probability = analysis?.prediction.vessels[v]?.probability;
    return probability == null ? neutral : new Color(riskColor(probability));
  }), [analysis]);
  useEffect(() => { transition.current = 1; invalidate(); }, [mode, analysis, selected, hovered, mapping, invalidate]);
  useEffect(() => {
    const [x, y, z] = positions[view];
    void controls.current?.setLookAt(x, y, z, 0, 0, 0, true);
  }, [view, reset]);
  const anchors = useMemo(() => {
    const best = vessels.map(() => ({ score: -Infinity, point: new Vector3() }));
    const vertex = new Vector3(), offset = new Vector3();
    shell.traverse(o => {
      if (!(o instanceof Mesh)) return;
      const position = o.geometry.getAttribute('position');
      for (let j = 0; j < position.count; j++) {
        vertex.fromBufferAttribute(position, j).applyMatrix4(o.matrixWorld);
        if (vertex.y > frame.band.x) continue;
        offset.copy(vertex).sub(frame.center);
        const i = territoryIndex(offset.clone().normalize(), mapping.axes);
        if (i < 0) continue;
        const score = offset.dot(labelSide[vessels[i]]);
        if (score > best[i].score) { best[i].score = score; best[i].point.copy(vertex); }
      }
    });
    const result = new Map<Vessel, Vector3>();
    best.forEach(({ score, point }, i) => { if (score > -Infinity) result.set(vessels[i], point.addScaledVector(labelSide[vessels[i]], 0.08)); });
    return result;
  }, [shell, frame, mapping]);
  const labelRefs = useRef(new Map<Vessel, HTMLButtonElement>());
  const toCamera = useMemo(() => new Vector3(), []), facing = useMemo(() => new Vector3(), []);
  const focus = useMemo(() => new Vector3(), []);
  useEffect(() => {
    if (!selected) return;
    const anchor = anchors.get(selected);
    if (!anchor) return;
    // Turn toward the territory's side, aiming between it and the heart's center so the whole heart stays framed.
    const target = anchor.clone().multiplyScalar(0.5);
    const direction = anchor.clone().setY(0);
    if (direction.length() < 0.1) direction.set(0, 0, 1);
    direction.normalize();
    const eye = target.clone().addScaledVector(direction, 3.8); eye.y += 0.3;
    void controls.current?.setLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, true);
  }, [selected, anchors]);
  useFrame(({ camera }, delta) => {
    // Hide labels whose territory is on the far side of the heart from the camera.
    toCamera.copy(camera.position).normalize();
    for (const [vessel, element] of labelRefs.current) {
      const anchor = anchors.get(vessel);
      const visible = !anchor || facing.copy(anchor).sub(frame.center).normalize().dot(toCamera) > -0.2;
      element.style.visibility = visible ? 'visible' : 'hidden';
    }
    if (group.current) {
      if (playing) phase.current += Math.min(delta, 0.1) * bpm / 60 * Math.PI * 2;
      const beat = playing ? Math.pow(Math.max(0, Math.sin(phase.current)), 4) * 0.018 : 0;
      group.current.scale.set(1 - beat, 1 + beat * 0.45, 1 - beat);
    }
    const ease = 1 - Math.exp(-delta * 9);
    uniforms.uRiskStrength.value += ((risk ? 1 : 0) - uniforms.uRiskStrength.value) * ease;
    colors.forEach((color, i) => uniforms.uRiskColors.value[i].lerp(color, ease));
    const emphasis = (v: Vessel) => v === selected ? 1 : v === hovered ? 0.6 : 0;
    uniforms.uRiskFocus.value.lerp(focus.set(emphasis('LAD'), emphasis('LCX'), emphasis('RCA')), ease);
    if (transition.current > 0) { transition.current -= delta; invalidate(); }
  });
  /** The vessel whose territory holds this surface point, or nothing over the atria and great vessels. */
  const territoryAt = (point: Vector3): Vessel | undefined => {
    if (point.y > frame.band.y) return undefined;
    return vessels[territoryIndex(point.clone().sub(frame.center).normalize(), mapping.axes)];
  };
  const vesselEvent = (event: ThreeEvent<PointerEvent | MouseEvent>) => {
    const vessel = risk ? territoryAt(event.point) : undefined;
    if (vessel) event.stopPropagation();
    return vessel;
  };
  return <>
    <CameraControls ref={controls} minDistance={1.7} maxDistance={7} smoothTime={0.3} {...touchControls}/>
    <group ref={group}>
      <primitive object={shell}
        onPointerMove={(e: ThreeEvent<PointerEvent>) => setHovered(vesselEvent(e) ?? null)}
        onPointerOut={() => setHovered(null)}
        onClick={(e: ThreeEvent<MouseEvent>) => { const vessel = vesselEvent(e); if (vessel) select(vessel); }}/>
      {labels && risk && [...anchors].map(([vessel, point]) => <Html key={vessel} position={point} center zIndexRange={[5, 0]} distanceFactor={4}><button ref={element => { if (element) labelRefs.current.set(vessel, element); else labelRefs.current.delete(vessel); }} className={`artery-label ${selected === vessel ? 'active' : ''}`} onClick={() => select(vessel)}>{vessel}</button></Html>)}
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
  const riskUnavailable = assetError ? 'Unavailable: the 3D model does not match the vessel registry' : !analysis ? 'Available once an estimate is ready' : undefined;
  // Territory labels mark the risk colors, so they are only offered in the risk overlay.
  const labelsAvailable = mode === 'risk' && !riskUnavailable;
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
        <Suspense fallback={null}><Heart playing={playing} bpm={bpm} labels={labels} view={view} reset={reset} onError={setAssetError}/></Suspense>
      </Canvas></SceneBoundary>
      <LoadingOverlay/>
      {assetError && <p className="asset-error" role="alert">Risk overlay disabled: {assetError}</p>}
    </div>
    <div className="viewer-bottom"><div className="view-presets">{(Object.keys(positions) as View[]).map(v => <button key={v} aria-pressed={view === v} onClick={() => { setView(v); setReset(n => n + 1); }}>{v}</button>)}</div><div className="view-tools"><button onClick={() => { setView('Anterior'); setReset(n => n + 1); useStore.getState().selectVessel(null); }}>↺ Reset</button><button aria-pressed={labels && labelsAvailable} disabled={!labelsAvailable} title={labelsAvailable ? undefined : 'Available in Risk overlay'} onClick={() => setLabels(v => !v)}>Labels</button><button aria-pressed={!paused && !reduced} disabled={reduced} title={reduced ? 'Paused by your reduced-motion setting' : undefined} onClick={() => setPaused(v => !v)}>{paused || reduced ? 'Play beat' : 'Pause beat'}</button></div></div>
    <div className="canvas-caption"><span>{coarsePointer ? 'Two fingers to rotate · pinch to zoom' : 'Drag to rotate · scroll to zoom'}</span><span>{playing ? `${bpm} BPM · illustrative motion` : 'Motion paused'}</span></div>
  </section>;
}
