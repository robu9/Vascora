"""Build the editable, representative heart from the pinned Z-Anatomy library.

Run with Blender 5.0.1's bpy module or: blender --background --python this_file.py.
Source geometry is never treated as patient-specific anatomy.
"""
import hashlib
import json
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)
with bpy.data.libraries.load(str(ROOT / 'source/heart-source.blend')) as (data_from, data_to):
    data_to.objects = [n for n in data_from.objects if n.removesuffix('__WORLD') in {
        'Left ventricle', 'Right ventricle', 'Left atrium', 'Right atrium',
        'Pulmonary trunk', 'Ascending aorta', 'Superior vena cava',
        'Anterior interventricular artery', 'Circumflex artery of heart',
        'Right coronary artery', 'Left coronary artery',
        'Right inferolateral branch of right coronary artery',
    }]
sources = {o.name.removesuffix('__WORLD'): o for o in data_to.objects if o}
scene = bpy.context.scene
out = []
source_map = {}

def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = .48
    return mat

tissue = material('Myocardium', (.29, .064, .048))
artery = material('Coronary anatomy', (.68, .32, .20))
root_mat = material('Great vessel roots', (.36, .12, .095))

def curve_part(name, specs, mat=artery):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 7
    curve.bevel_depth = .0005
    curve.bevel_resolution = 3
    curve.use_fill_caps = True
    for source_name, spline_index, start, end in specs:
        obj = sources[source_name]
        source = obj.data.splines[spline_index]
        points = list(source.bezier_points)[start:end]
        spline = curve.splines.new('BEZIER')
        spline.bezier_points.add(len(points) - 1)
        for p, original in zip(spline.bezier_points, points):
            p.co = obj.matrix_world @ original.co
            p.handle_left_type = 'FREE'; p.handle_right_type = 'FREE'
            p.handle_left = obj.matrix_world @ original.handle_left
            p.handle_right = obj.matrix_world @ original.handle_right
            p.radius = original.radius
    obj = bpy.data.objects.new(name, curve)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat.copy())
    out.append(obj)
    source_map[name] = specs
    return obj

for name in ['Left ventricle', 'Right ventricle', 'Left atrium', 'Right atrium', 'Pulmonary trunk']:
    src = sources[name]
    obj = bpy.data.objects.new(name.replace(' ', '_'), src.data.copy())
    obj.matrix_world = src.matrix_world.copy()
    scene.collection.objects.link(obj)
    obj.data.materials.clear(); obj.data.materials.append(tissue if name != 'Pulmonary trunk' else root_mat)
    subdivision = obj.modifiers.new('Surface smoothing', 'SUBSURF'); subdivision.levels = 1
    out.append(obj)

lad = 'Anterior interventricular artery'; lcx = 'Circumflex artery of heart'; rca = 'Right coronary artery'
curve_part('LAD_prox', [(lad, 0, 0, 5)])
curve_part('LAD_mid', [(lad, 0, 4, 9), (lad, 4, 0, None), (lad, 5, 0, None)])
curve_part('LAD_dist', [(lad, 0, 8, None), (lad, 1, 0, None)])
curve_part('D1', [(lad, 2, 0, None), (lad, 3, 0, None)])
curve_part('LCX_prox', [(lcx, 0, 0, 7)])
curve_part('LCX_mid', [(lcx, 0, 6, 14)])
curve_part('LCX_dist', [(lcx, 0, 13, None)] + [(lcx, i, 0, None) for i in [1, 2, 4, 5]])
curve_part('OM1', [(lcx, 3, 0, None)])
curve_part('RCA_prox', [(rca, 0, 0, 6)] + [(rca, i, 0, None) for i in [9, 10, 11, 12, 13]])
curve_part('RCA_mid', [(rca, 0, 5, 11)] + [(rca, i, 0, None) for i in range(1, 9)])
curve_part('RCA_dist', [(rca, 0, 10, None)] + [('Right inferolateral branch of right coronary artery', i, 0, None) for i in range(8)])
curve_part('Left_main', [('Left coronary artery', 0, 0, None)], root_mat)
curve_part('Aorta_root', [('Ascending aorta', 0, 0, None)], root_mat)
curve_part('SVC_root', [('Superior vena cava', 0, 0, None)], root_mat)

# No explicitly identified PDA is present in this source. Author a representative
# right-dominant PDA, connected to the RCA's native posterior endpoint. Its path
# follows the inferior interventricular region and is fitted to the outer surface.
curve = bpy.data.curves.new('PDA', 'CURVE'); curve.dimensions = '3D'
curve.bevel_depth = .0005; curve.bevel_resolution = 3; curve.resolution_u = 7; curve.use_fill_caps = True
spline = curve.splines.new('BEZIER')
points = [(.0215631, .00412175, 1.2570221), (.026, -.005, 1.25), (.032, -.018, 1.244), (.04, -.032, 1.24), (.048, -.046, 1.24), (.0572, -.0624, 1.244)]
spline.bezier_points.add(len(points) - 1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
surfaces = [o for o in out if o.name in {'Left_ventricle', 'Right_ventricle'}]
for i, (p, co) in enumerate(zip(spline.bezier_points, points)):
    fitted = Vector(co)
    if 0 < i < len(points) - 1:
        hits = []
        for surface in surfaces:
            evaluated = surface.evaluated_get(depsgraph)
            inv = evaluated.matrix_world.inverted()
            hit, location, _, _ = evaluated.ray_cast(inv @ Vector((co[0], co[1], 1.1)), inv.to_3x3() @ Vector((0, 0, 1)))
            if hit: hits.append(evaluated.matrix_world @ location)
        if hits: fitted = min(hits, key=lambda v: v.z) - Vector((0, 0, .0006))
    p.co = fitted; p.handle_left_type = 'AUTO'; p.handle_right_type = 'AUTO'; p.radius = 1.4 - i * .17
pda = bpy.data.objects.new('PDA', curve); scene.collection.objects.link(pda); pda.data.materials.append(artery.copy()); out.append(pda)
source_map['PDA'] = 'Authored representative right-dominant branch; source has no explicitly named PDA.'

# Convert evaluated surfaces and curves; apply one common normalization. Blender
# source is X=patient left, -Y=anterior, Z=superior; glTF converts to Y-up.
for obj in out:
    bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target='MESH')
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for polygon in obj.data.polygons: polygon.use_smooth = True

minimum = Vector((float('inf'),)*3); maximum = Vector((float('-inf'),)*3)
for obj in out:
    for v in obj.data.vertices:
        p = obj.matrix_world @ v.co
        for axis in range(3): minimum[axis] = min(minimum[axis], p[axis]); maximum[axis] = max(maximum[axis], p[axis])
center = (minimum + maximum) / 2
scale = 2.25 / max(maximum - minimum)
for obj in out:
    transform = obj.matrix_world.copy()
    for v in obj.data.vertices: v.co = (transform @ v.co - center) * scale
    obj.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.normals_make_consistent(inside=False); bpy.ops.object.mode_set(mode='OBJECT')

# Remove source-only datablocks and text scripts from the delivered editable file.
for obj in list(bpy.data.objects):
    if obj not in out: bpy.data.objects.remove(obj, do_unlink=True)
for text in list(bpy.data.texts): bpy.data.texts.remove(text)
bpy.ops.outliner.orphans_purge(do_recursive=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / 'heart.blend'))
bpy.ops.export_scene.gltf(filepath=str(ROOT / 'heart.raw.glb'), export_format='GLB', use_selection=True, export_yup=True, export_apply=True, export_materials='EXPORT')
archive = ROOT / 'source/Z-Anatomy.zip'
manifest = {
    'source_url': 'https://raw.githubusercontent.com/Z-Anatomy/Models-of-human-anatomy/master/Z-Anatomy.zip',
    'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'blender_version': bpy.app.version_string,
    'normalization': {'source_center': list(center), 'scale': scale, 'gltf_axes': '+X patient left; +Y superior; +Z anterior'},
    'meshes': source_map,
    'anatomical_limitations': ['Representative educational anatomy, not patient-specific.', 'Proximal/mid/distal divisions are visualization partitions, not standardized clinical lesion segments.', 'D1 and OM1 labels designate representative source branches; numbering is not clinically validated.', 'PDA is an authored surface-fitted right-dominant approximation; source does not explicitly identify a PDA.'],
}
(ROOT / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Exported', len(out), 'objects', flush=True)
