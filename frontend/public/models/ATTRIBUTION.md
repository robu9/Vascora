# Heart model attribution

The heart shown in the viewer is **Realistic Human Heart** by neshallads, used unmodified
under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Commercial use is allowed. Credit: neshallads,
https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089.
The file `realistic_human_heart.glb` is the Sketchfab glTF binary with 2k textures.

`heart.glb` remains the Z-Anatomy / BodyParts3D adaptation described below. The viewer
loads it to check the vessel-name contract and to size the new heart. Those meshes are
not drawn, because they were built for a different shape. That file stays under
CC BY-SA 4.0, which does not replace the Attribution terms on the Sketchfab surface.

The coronary adaptation comes from:

- **Z-Anatomy — The libre 3D atlas of anatomy — CC BY-SA 4.0**.
  Gauthier Kervyn (design, 3D, anatomy), Marcin Zielinski (Blender add-on).
  https://github.com/Z-Anatomy/Models-of-human-anatomy
- **BodyParts3D — The Database Center for Life Science — CC BY-SA 2.1 Japan**.
  Kousaku Okubo (original model).
  https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html

Vascora adaptations: extraction of external heart surfaces and coronary curves,
surface smoothing, representative branch grouping, artery segmentation, authored
surface-fitted PDA, normalized coordinates, new materials, glTF export and meshopt
compression. Adapted model assets (`heart.glb` and the editable `heart.blend`) are
distributed under **Creative Commons Attribution-ShareAlike 4.0 International**:
https://creativecommons.org/licenses/by-sa/4.0/

The complete upstream notice is retained in `LICENSE-source.txt`. Its references
to unrelated organs do not indicate those organs are included in this heart model.
No upstream executable add-ons or anatomical definition texts are included.

## Anatomical interpretation

This is representative educational anatomy, not a patient reconstruction or a
clinically validated anatomical atlas. Proximal/mid/distal names are visualization
partitions. D1 and OM1 are representative source branch assignments. The upstream
heart does not explicitly identify a PDA: Vascora authors a right-dominant,
surface-fitted approximation connected to the RCA posterior endpoint. This does
not establish the source donor's coronary dominance.

Risk colors apply equally throughout each vessel group. They do not indicate
plaque position, narrowing percentage, measured blood flow, or regional dysfunction.

Source SHA-256, branch mappings and reproduction details are in the repository's
`assets-pipeline/provenance.json` and `assets-pipeline/README.md`.
