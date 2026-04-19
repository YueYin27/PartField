"""
Downsample a multi-part GLB mesh while preserving part structure.

Each part (named geometry) in the GLB scene graph is decimated independently
via pymeshlab's quadric edge collapse. Node names, hierarchy, transforms,
and materials are preserved.

Usage:
    python downsample_mesh.py --input segmented.glb --output segmented_low.glb --target_faces 5000
    python downsample_mesh.py --input segmented.glb --output segmented_low.glb --ratio 0.5
    python downsample_mesh.py --input_dir meshes/ --output_dir meshes_low/ --target_faces 10000
"""

import argparse
import os
import glob
import numpy as np
import pymeshlab
import trimesh


def decimate_trimesh(mesh, target_faces=None, ratio=None):
    """Decimate a single trimesh.Trimesh via pymeshlab, returning a new Trimesh."""
    original_faces = len(mesh.faces)

    if target_faces is not None:
        target = target_faces
    elif ratio is not None:
        target = max(1, int(original_faces * ratio))
    else:
        raise ValueError("Specify either target_faces or ratio")

    if target >= original_faces:
        print(f"    Target ({target}) >= current faces ({original_faces}), keeping as-is.")
        return mesh

    # pymeshlab's quadric decimation uses absolute-tolerance heuristics that
    # can collapse tiny meshes (e.g. 0.05 unit diameter) into garbage. Scale
    # the mesh to ~unit diameter around its centroid before decimation and
    # invert the scale afterward.
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = verts.mean(axis=0)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    scale = 1.0
    if diag > 0 and diag < 1.0:
        scale = 1.0 / diag
    verts_scaled = (verts - center) * scale

    ml_mesh = pymeshlab.Mesh(
        vertex_matrix=verts_scaled,
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(ml_mesh, "part")

    # GLB meshes commonly store duplicated vertices (one per triangle corner)
    # to support per-face normals/UVs. Quadric decimation treats each triangle
    # as topologically isolated in that case and cannot merge edges, producing
    # a sparse "disconnected triangles" result. Merge coincident vertices and
    # clean up before decimating.
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_merge_close_vertices(
        threshold=pymeshlab.PercentageValue(0.0001)
    )
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target, preservenormal=True
    )
    processed = ms.current_mesh()
    new_verts = processed.vertex_matrix()
    new_faces = processed.face_matrix()

    # Invert the pre-decimation scale/translation.
    new_verts = new_verts / scale + center

    new_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=new_faces, process=False
    )
    try:
        new_mesh.visual = mesh.visual.copy()
    except Exception:
        pass
    return new_mesh


def downsample_glb(input_path, output_path, target_faces=None, ratio=None,
                   total_faces=None):
    """Decimate every geometry in a GLB scene, preserving the scene graph."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loaded = trimesh.load(input_path, process=False)

    # Normalize to a Scene so single-geometry GLBs are handled the same way.
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded, geom_name="geometry_0", node_name="geometry_0")
    else:
        scene = loaded

    # Some GLBs (e.g. apple.glb) store per-node rotations in the scene graph
    # instead of baking them into the vertices. We bake those transforms into
    # fresh mesh copies and rebuild the scene with identity node transforms so
    # that every export path (including re-decimation) is unambiguous.
    baked = trimesh.Scene()
    node_names = list(scene.graph.nodes_geometry)
    for node_name in node_names:
        transform, geom_name = scene.graph[node_name]
        if geom_name is None or geom_name not in scene.geometry:
            continue
        src = scene.geometry[geom_name]
        # Deep-copy vertices/faces explicitly to avoid any shared-array issues.
        geom = trimesh.Trimesh(
            vertices=np.array(src.vertices, dtype=np.float64, copy=True),
            faces=np.array(src.faces, dtype=np.int64, copy=True),
            process=False,
        )
        try:
            geom.visual = src.visual.copy()
        except Exception:
            pass
        if not np.allclose(transform, np.eye(4)):
            geom.apply_transform(np.asarray(transform, dtype=np.float64))
        baked.add_geometry(
            geom, geom_name=node_name, node_name=node_name,
            transform=np.eye(4),
        )
    scene = baked

    print(f"Loaded: {input_path}")
    print(f"  Parts: {len(scene.geometry)}")
    total_faces_before = sum(len(g.faces) for g in scene.geometry.values())
    print(f"  Total faces: {total_faces_before}")

    # If a global face budget was given, allocate per-part quotas in proportion
    # to each part's original face count (largest-remainder for rounding).
    per_part_target = {}
    if total_faces is not None:
        if total_faces >= total_faces_before:
            print(f"  --total_faces ({total_faces}) >= current total "
                  f"({total_faces_before}); keeping parts as-is.")
            for name, geom in scene.geometry.items():
                per_part_target[name] = len(geom.faces)
        else:
            counts = {n: len(g.faces) for n, g in scene.geometry.items()}
            raw = {n: total_faces * c / total_faces_before for n, c in counts.items()}
            floored = {n: max(1, int(v)) for n, v in raw.items()}
            # Distribute remaining budget by largest fractional part.
            remaining = total_faces - sum(floored.values())
            order = sorted(raw.items(), key=lambda kv: (kv[1] - int(kv[1])),
                           reverse=True)
            i = 0
            while remaining > 0 and order:
                name = order[i % len(order)][0]
                if floored[name] < counts[name]:
                    floored[name] += 1
                    remaining -= 1
                i += 1
                if i > 10 * len(order):
                    break
            per_part_target = floored

    # Decimate each geometry in place. scene.geometry maps name -> Trimesh,
    # and scene.graph references geometries by name, so replacing the entry
    # keeps node names, transforms, and hierarchy intact.
    new_geometry = {}
    for name, geom in scene.geometry.items():
        if total_faces is not None:
            tgt = per_part_target[name]
            print(f"  Decimating part '{name}' ({len(geom.faces)} -> {tgt} faces)")
            new_geometry[name] = decimate_trimesh(geom, target_faces=tgt)
        else:
            print(f"  Decimating part '{name}' ({len(geom.faces)} faces)")
            new_geometry[name] = decimate_trimesh(
                geom, target_faces=target_faces, ratio=ratio
            )

    # Replace geometries while preserving the scene graph.
    scene.geometry.clear()
    for name, geom in new_geometry.items():
        scene.geometry[name] = geom

    total_faces_after = sum(len(g.faces) for g in scene.geometry.values())
    print(f"  Total faces after: {total_faces_after}")

    # trimesh always wraps geometries under a synthetic "world" root node.
    # Remove it entirely and reindex so part nodes sit at the scene root.
    def _strip_world_root(tree):
        nodes = tree.get("nodes", [])
        if not nodes or nodes[0].get("name") != "world":
            return
        world_children = nodes[0].get("children", [])
        # Drop node 0 ("world") and shift every remaining index down by 1.
        new_nodes = nodes[1:]
        for n in new_nodes:
            if "children" in n:
                n["children"] = [c - 1 for c in n["children"]]
        tree["nodes"] = new_nodes
        # Point every scene at the (reindexed) former children of "world".
        new_scene_roots = [c - 1 for c in world_children]
        for scene in tree.get("scenes", []):
            scene["nodes"] = list(new_scene_roots)

    glb_bytes = trimesh.exchange.gltf.export_glb(
        scene, tree_postprocessor=_strip_world_root
    )
    with open(output_path, "wb") as f:
        f.write(glb_bytes)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample a multi-part GLB mesh, preserving part names and structure."
    )
    parser.add_argument("--input", type=str, help="Input GLB file")
    parser.add_argument("--output", type=str, help="Output GLB file")
    parser.add_argument("--input_dir", type=str, help="Input directory for batch processing")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch processing")
    parser.add_argument("--target_faces", type=int, default=None,
                        help="Target number of faces PER PART")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Ratio of faces to keep per part (0.0-1.0)")
    parser.add_argument("--total_faces", type=int, default=None,
                        help="Total face budget across all parts (proportional)")

    args = parser.parse_args()

    specified = sum(x is not None for x in
                    (args.target_faces, args.ratio, args.total_faces))
    if specified == 0:
        parser.error("Specify one of --target_faces, --ratio, or --total_faces")
    if specified > 1:
        parser.error("Specify only one of --target_faces, --ratio, --total_faces")

    def _check_glb(path):
        if not path.lower().endswith(".glb"):
            raise ValueError(f"Only .glb inputs/outputs are supported, got: {path}")

    if args.input_dir:
        if not args.output_dir:
            parser.error("--output_dir is required with --input_dir")
        os.makedirs(args.output_dir, exist_ok=True)

        glb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.glb")))
        print(f"Found {len(glb_files)} GLB files in {args.input_dir}")
        for fpath in glb_files:
            fname = os.path.basename(fpath)
            out_path = os.path.join(args.output_dir, fname)
            try:
                downsample_glb(fpath, out_path,
                               target_faces=args.target_faces,
                               ratio=args.ratio,
                               total_faces=args.total_faces)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    else:
        if not args.input or not args.output:
            parser.error("Specify --input and --output for single file mode")
        _check_glb(args.input)
        _check_glb(args.output)
        downsample_glb(args.input, args.output,
                       target_faces=args.target_faces,
                       ratio=args.ratio,
                       total_faces=args.total_faces)
