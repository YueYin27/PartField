"""
Convert a colored PLY or GLB mesh into a USDZ file with separate named parts.

Faces sharing the same color are grouped into one part. Each part becomes a
separate USD prim named part1, part2, ...

Usage:
    python mesh_to_usdz.py --input segmented.ply --output model.usdz
    python mesh_to_usdz.py --input segmented.glb --output model.usdz
    python mesh_to_usdz.py --input_dir exp_results/clustering_final/ --output_dir exp_results/usdz/
"""

import argparse
import os
import glob
import numpy as np
import trimesh

from pxr import Usd, UsdGeom, Vt, Gf, UsdUtils


def load_mesh_and_face_labels(input_path):
    """
    Load a mesh and extract per-face part labels from face colors.

    Faces with the same RGB color are assigned the same label.

    Returns
    -------
    vertices : np.ndarray (V, 3)
    faces : np.ndarray (F, 3)
    face_labels : np.ndarray (F,) integer label per face
    """
    mesh = trimesh.load(input_path, force='mesh', process=False)

    # Get per-face colors
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        face_colors = np.array(mesh.visual.face_colors)[:, :3]  # drop alpha
    else:
        face_colors = np.zeros((len(mesh.faces), 3), dtype=np.uint8)

    # Map unique colors to integer labels
    color_to_label = {}
    face_labels = np.zeros(len(mesh.faces), dtype=int)
    label_counter = 0

    for i, color in enumerate(face_colors):
        key = tuple(color)
        if key not in color_to_label:
            color_to_label[key] = label_counter
            label_counter += 1
        face_labels[i] = color_to_label[key]

    return np.array(mesh.vertices), np.array(mesh.faces), face_labels


def split_mesh_by_labels(vertices, faces, face_labels):
    """
    Split a mesh into sub-meshes based on face labels.

    Returns
    -------
    parts : list of (part_vertices, part_faces) tuples.
            Faces are re-indexed to the local vertex array.
    """
    unique_labels = np.unique(face_labels)
    parts = []

    for label in unique_labels:
        mask = face_labels == label
        part_faces_global = faces[mask]

        unique_verts, inverse = np.unique(part_faces_global.ravel(), return_inverse=True)
        part_vertices = vertices[unique_verts]
        part_faces = inverse.reshape(-1, 3)

        parts.append((part_vertices, part_faces))

    return parts


def build_usdz(parts, output_path):
    """
    Build a USDZ file with each part as a separate named prim.

    Parameters
    ----------
    parts : list of (vertices, faces) tuples
    output_path : str
    """
    usdc_path = output_path.replace('.usdz', '_tmp.usdc')

    stage = Usd.Stage.CreateNew(usdc_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, '/Root')

    for i, (verts, faces) in enumerate(parts):
        part_name = f'part{i + 1}'
        mesh_path = f'/Root/{part_name}'

        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts]
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        face_counts = Vt.IntArray([3] * len(faces))
        usd_mesh.GetFaceVertexCountsAttr().Set(face_counts)

        indices = Vt.IntArray(faces.ravel().tolist())
        usd_mesh.GetFaceVertexIndicesAttr().Set(indices)

    stage.GetRootLayer().Save()

    UsdUtils.CreateNewUsdzPackage(usdc_path, output_path)
    os.remove(usdc_path)

    print(f"Exported USDZ: {output_path} ({len(parts)} parts)")


def convert_mesh_to_usdz(input_path, output_path):
    """Full pipeline: load colored mesh -> split by color -> export USDZ."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    vertices, faces, face_labels = load_mesh_and_face_labels(input_path)
    n_parts = len(np.unique(face_labels))
    print(f"Loaded: {input_path} ({len(vertices)} verts, {len(faces)} faces, {n_parts} parts)")

    parts = split_mesh_by_labels(vertices, faces, face_labels)
    build_usdz(parts, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert colored PLY/GLB to USDZ with named parts.")
    parser.add_argument('--input', type=str, help='Input mesh file (ply or glb)')
    parser.add_argument('--output', type=str, help='Output usdz file')
    parser.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch processing')

    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            parser.error("--output_dir is required with --input_dir")
        os.makedirs(args.output_dir, exist_ok=True)

        mesh_files = []
        for ext in ('*.ply', '*.glb'):
            mesh_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        mesh_files.sort()

        print(f"Found {len(mesh_files)} meshes in {args.input_dir}")
        for fpath in mesh_files:
            fname = os.path.splitext(os.path.basename(fpath))[0] + '.usdz'
            out_path = os.path.join(args.output_dir, fname)
            try:
                convert_mesh_to_usdz(fpath, out_path)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
    else:
        if not args.input or not args.output:
            parser.error("Specify --input and --output for single file mode")
        convert_mesh_to_usdz(args.input, args.output)
