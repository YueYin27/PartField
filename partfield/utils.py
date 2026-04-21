import numpy as np
import trimesh

# Rotation that undoes the Z-up -> Y-up conversion applied when exporting GLB.
# remesh.py bakes a -90deg rotation about X into the GLB so that Y-up viewers
# display it matching the original Z-up PLY. Here we rotate back to Z-up so
# the mesh we feed into PartField matches the source PLY's frame.
# Mapping: (x, y, z)_Yup -> (x, -z, y)_Zup  (a +90deg rotation about X).
_YUP_TO_ZUP = np.array([
    [1, 0,  0, 0],
    [0, 0, -1, 0],
    [0, 1,  0, 0],
    [0, 0,  0, 1],
], dtype=np.float64)


def load_mesh_util(input_fname):
    mesh = trimesh.load(input_fname, force='mesh', process=False)
    if input_fname.lower().endswith('.glb'):
        mesh.apply_transform(_YUP_TO_ZUP)
    return mesh