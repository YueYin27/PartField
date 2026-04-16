"""
Split a segmented mesh into one PLY file per part.

Usage:
    python split_parts.py --mesh exp_results/partfield_features/mesh/train/input_train_0.ply \
                          --labels exp_results/clustering/mesh/train/cluster_out/train_0_05.npy \
                          --out_dir exp_results/parts/train_05
"""

import argparse
import os
import numpy as np
import trimesh


def split_parts(mesh_path, labels_path, out_dir):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    labels = np.load(labels_path).flatten().astype(int)

    assert len(labels) == len(mesh.faces), (
        f"Label count ({len(labels)}) != face count ({len(mesh.faces)})"
    )

    os.makedirs(out_dir, exist_ok=True)

    unique_parts = np.unique(labels)
    print(f"Mesh has {len(mesh.faces)} faces, {len(unique_parts)} parts")

    for idx, part_id in enumerate(unique_parts):
        face_mask = labels == part_id
        part_faces = mesh.faces[face_mask]

        # Remap vertices to only those used by this part
        used_verts, inverse = np.unique(part_faces, return_inverse=True)
        new_faces = inverse.reshape(part_faces.shape)
        new_verts = mesh.vertices[used_verts]

        part_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        out_path = os.path.join(out_dir, f"part_{idx:02d}.ply")
        part_mesh.export(out_path)
        print(f"  Part {idx}: {face_mask.sum()} faces -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to input mesh PLY (input_<uid>_0.ply)")
    parser.add_argument("--labels", required=True, help="Path to face-label .npy file from cluster_out/")
    parser.add_argument("--out_dir", required=True, help="Directory to write per-part PLY files")
    args = parser.parse_args()

    split_parts(args.mesh, args.labels, args.out_dir)
