"""
Load a PLY mesh, remesh it using MeshLab's isotropic remeshing filter via pymeshlab,
and export the result as a GLB file.

Usage:
    python remesh.py --input mesh.ply --output remeshed.glb
    python remesh.py --input mesh.ply --output remeshed.glb --target_faces 5000
    python remesh.py --input mesh.ply --output remeshed.glb --iterations 5 --adaptive
"""

import argparse
import os
import numpy as np
import trimesh
import pymeshlab


def remesh_ply_to_glb(
    input_path: str,
    output_path: str,
    target_faces: int = None,
    iterations: int = 3,
    adaptive: bool = False,
):
    """
    Load a PLY mesh, apply isotropic remeshing via MeshLab, and save as GLB.

    Parameters
    ----------
    input_path : str
        Path to the input PLY file.
    output_path : str
        Path to the output GLB file.
    target_faces : int, optional
        Target number of faces after remeshing. If None, MeshLab uses its
        default target edge length (computed from the mesh bounding box).
    iterations : int
        Number of remeshing iterations (default: 3).
    adaptive : bool
        If True, use adaptive remeshing (non-uniform edge lengths based on
        local curvature). If False, use uniform isotropic remeshing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load and remesh with pymeshlab
    # -------------------------------------------------------------------------
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    print(f"Loaded: {input_path}")
    print(f"  Vertices : {ms.current_mesh().vertex_number()}")
    print(f"  Faces    : {ms.current_mesh().face_number()}")

    if target_faces is not None:
        # Compute an approximate target edge length from the desired face count.
        # For a uniform triangulation the relationship is roughly:
        #   area ≈ (sqrt(3)/4) * edge_len^2 * n_faces
        # We use the bounding-box diagonal as a proxy for the mesh surface area.
        bbox = ms.current_mesh().bounding_box()
        diag = bbox.diagonal()
        # Heuristic: scale edge length so that the remeshed mesh has ~target_faces faces.
        # surface_area ≈ diag^2 * 0.5 (rough approximation)
        import math
        approx_area = (diag ** 2) * 0.5
        target_edge_len = math.sqrt((4 * approx_area) / (math.sqrt(3) * target_faces))
        print(f"  Target faces   : {target_faces}")
        print(f"  Target edge len: {target_edge_len:.6f}")
    else:
        target_edge_len = None

    remesh_kwargs = dict(iterations=iterations)
    if target_edge_len is not None:
        remesh_kwargs["targetlen"] = pymeshlab.PercentageValue(target_edge_len)

    if adaptive:
        print("Applying adaptive remeshing...")
        ms.meshing_isotropic_explicit_remeshing(
            adaptive=True,
            **remesh_kwargs,
        )
    else:
        print("Applying isotropic remeshing...")
        ms.meshing_isotropic_explicit_remeshing(**remesh_kwargs)

    print(f"Remeshed:")
    print(f"  Vertices : {ms.current_mesh().vertex_number()}")
    print(f"  Faces    : {ms.current_mesh().face_number()}")

    # -------------------------------------------------------------------------
    # 2. Export to a temporary PLY, then convert to GLB via trimesh
    #    (pymeshlab can write PLY/OBJ natively; trimesh handles GLB export)
    # -------------------------------------------------------------------------
    tmp_ply = output_path.replace(".glb", "_tmp.ply")
    ms.save_current_mesh(tmp_ply)

    mesh = trimesh.load(tmp_ply, force="mesh", process=False)

    # glTF/GLB is Y-up by spec; most PLY files are Z-up. Rotate Z-up -> Y-up
    # so the GLB displays in a glTF viewer the same way the PLY displays in
    # a Z-up viewer. This is -90deg about the X axis: (x, y, z) -> (x, z, -y).
    zup_to_yup = trimesh.transformations.rotation_matrix(
        angle=-np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
    )
    mesh.apply_transform(zup_to_yup)

    mesh.export(output_path)
    print(f"Exported GLB: {output_path}")

    os.remove(tmp_ply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remesh a PLY file using MeshLab and export as GLB."
    )
    parser.add_argument("--input", required=True, type=str, help="Input PLY file path")
    parser.add_argument("--output", required=True, type=str, help="Output GLB file path")
    parser.add_argument(
        "--target_faces",
        default=None,
        type=int,
        help="Target number of faces after remeshing (optional)",
    )
    parser.add_argument(
        "--iterations",
        default=3,
        type=int,
        help="Number of remeshing iterations (default: 3)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive (curvature-aware) remeshing instead of uniform isotropic",
    )

    args = parser.parse_args()

    remesh_ply_to_glb(
        input_path=args.input,
        output_path=args.output,
        target_faces=args.target_faces,
        iterations=args.iterations,
        adaptive=args.adaptive,
    )
