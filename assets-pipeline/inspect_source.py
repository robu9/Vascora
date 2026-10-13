"""Inspect only the heart collection; never execute the source's bundled add-ons."""
import json
from pathlib import Path
import bpy

root = Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(root / 'source/Z-Anatomy/Startup.blend'), load_ui=False)
collections = [c for c in bpy.data.collections if any(t in c.name.lower() for t in ['heart', 'cardia'])]
print('COLLECTIONS', [(c.name, len(c.all_objects)) for c in collections], flush=True)
heart = set()
for c in collections:
    if c.name in {'Heart', 'Arteries of heart'}:
        heart.update(c.all_objects)
for o in bpy.data.objects:
    if o.name in {'Ascending aorta', 'Arch of aorta', 'Pulmonary trunk', 'Superior vena cava', 'Inferior vena cava'}:
        heart.add(o)
report = []
for o in sorted(heart, key=lambda o: o.name):
    if o.type not in {'MESH', 'CURVE'}:
        continue
    row = {'name': o.name, 'type': o.type, 'bounds': [list(o.matrix_world @ __import__('mathutils').Vector(p)) for p in o.bound_box]}
    if o.type == 'CURVE':
        row['bevel_depth'] = o.data.bevel_depth
        row['splines'] = [{'type': s.type, 'points': [list(o.matrix_world @ p.co.to_3d()) for p in (s.bezier_points if s.type == 'BEZIER' else s.points)], 'radii': [p.radius for p in (s.bezier_points if s.type == 'BEZIER' else s.points)]} for s in o.data.splines]
    report.append(row)
(root / 'source-inventory.json').write_text(json.dumps(report, indent=2))
print('HEART OBJECTS', [(r['name'], r['type']) for r in report], flush=True)
isolated = set()
for original in sorted(heart, key=lambda o: o.name):
    if original.type not in {'MESH', 'CURVE'}:
        continue
    data = original.data.copy()
    data.transform(original.matrix_world)
    data.materials.clear()
    obj = bpy.data.objects.new(original.name + '__WORLD', data)
    isolated.add(obj)
bpy.data.libraries.write(str(root / 'source/heart-source.blend'), isolated, fake_user=True)
