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


def _collapse_shortest_edge(mesh):
    """Merge the two endpoints of the shortest edge, removing exactly 1 vertex.
    Degenerate faces (those that become duplicate-vertex triangles) are dropped."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    edges = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) == 0:
        return mesh

    lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    v0, v1 = edges[int(np.argmin(lengths))]

    new_V = V.copy()
    new_V[v0] = (V[v0] + V[v1]) / 2.0
    new_F = F.copy()
    new_F[new_F == v1] = v0
    keep = ((new_F[:, 0] != new_F[:, 1]) &
            (new_F[:, 1] != new_F[:, 2]) &
            (new_F[:, 0] != new_F[:, 2]))
    new_F = new_F[keep]

    mask = np.ones(len(new_V), dtype=bool)
    mask[v1] = False
    remap = np.cumsum(mask) - 1
    new_V = new_V[mask]
    new_F = remap[new_F]

    out = trimesh.Trimesh(vertices=new_V, faces=new_F, process=False)
    try:
        out.visual = mesh.visual.copy()
    except Exception:
        pass
    return out


def _split_longest_edge(mesh):
    """Insert a midpoint on the longest edge, adding exactly 1 vertex."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if len(F) == 0:
        return mesh

    e_lens = np.stack([
        np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1),
        np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1),
        np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1),
    ], axis=1)
    flat = int(np.argmax(e_lens))
    face_idx, edge_idx = divmod(flat, 3)
    a, b, c = F[face_idx]
    if edge_idx == 0:
        v0, v1, other = a, b, c
    elif edge_idx == 1:
        v0, v1, other = b, c, a
    else:
        v0, v1, other = c, a, b

    v_new = len(V)
    new_V = np.vstack([V, ((V[v0] + V[v1]) / 2.0)[None, :]])

    edge_key = tuple(sorted((int(v0), int(v1))))
    new_faces = []
    for f in F:
        f0, f1, f2 = int(f[0]), int(f[1]), int(f[2])
        face_edges = [
            tuple(sorted((f0, f1))),
            tuple(sorted((f1, f2))),
            tuple(sorted((f2, f0))),
        ]
        if edge_key in face_edges:
            # Preserve winding: find the ccw cycle starting at v0.
            cycle = [f0, f1, f2]
            i0 = cycle.index(int(v0))
            a0 = cycle[i0]
            a1 = cycle[(i0 + 1) % 3]
            a2 = cycle[(i0 + 2) % 3]
            if a1 == int(v1):  # order v0 -> v1 -> other
                new_faces.append([a0, v_new, a2])
                new_faces.append([v_new, a1, a2])
            else:               # order v0 -> other -> v1
                new_faces.append([a0, a1, v_new])
                new_faces.append([v_new, a1, a2])  # a1 == other here
        else:
            new_faces.append([f0, f1, f2])

    out = trimesh.Trimesh(
        vertices=new_V,
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )
    try:
        out.visual = mesh.visual.copy()
    except Exception:
        pass
    return out


def _adjust_vertex_count(geometry_dict, target_verts):
    """Collapse/split edges on the largest part until total vertex count
    matches target_verts exactly. Mutates and returns geometry_dict."""
    for _ in range(200):
        total = sum(len(g.vertices) for g in geometry_dict.values())
        if total == target_verts:
            return geometry_dict
        # Operate on the part with the most faces for stability.
        name = max(geometry_dict, key=lambda n: len(geometry_dict[n].faces))
        if total > target_verts:
            geometry_dict[name] = _collapse_shortest_edge(geometry_dict[name])
        else:
            geometry_dict[name] = _split_longest_edge(geometry_dict[name])
    return geometry_dict


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
                   total_faces=None, total_vertices=None):
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
    total_verts_before = sum(len(g.vertices) for g in scene.geometry.values())
    print(f"  Total faces: {total_faces_before}  Total vertices: {total_verts_before}")

    def _allocate_per_part(face_budget):
        """Return {name: target_faces} proportional to original face counts."""
        if face_budget >= total_faces_before:
            return {name: len(geom.faces) for name, geom in scene.geometry.items()}
        counts = {n: len(g.faces) for n, g in scene.geometry.items()}
        raw = {n: face_budget * c / total_faces_before for n, c in counts.items()}
        floored = {n: max(1, int(v)) for n, v in raw.items()}
        remaining = face_budget - sum(floored.values())
        order = sorted(raw.items(), key=lambda kv: (kv[1] - int(kv[1])), reverse=True)
        i = 0
        while remaining > 0 and order:
            name = order[i % len(order)][0]
            if floored[name] < counts[name]:
                floored[name] += 1
                remaining -= 1
            i += 1
            if i > 10 * len(order):
                break
        return floored

    def _run_decimation(per_part_target):
        result = {}
        for name, geom in scene.geometry.items():
            tgt = per_part_target[name]
            print(f"  Decimating part '{name}' ({len(geom.faces)} -> {tgt} faces)")
            result[name] = decimate_trimesh(geom, target_faces=tgt)
        return result

    # Convert vertex budget to face budget using this mesh's V/F ratio,
    # then iterate to hit the target vertex count precisely.
    if total_vertices is not None:
        vf_ratio = total_verts_before / total_faces_before if total_faces_before > 0 else 0.5
        total_faces = max(1, int(round(total_vertices / vf_ratio)))
        print(f"  --total_vertices {total_vertices} → initial --total_faces {total_faces} "
              f"(V/F ratio {vf_ratio:.3f})")

        new_geometry = None
        for iteration in range(4):
            per_part_target = _allocate_per_part(total_faces)
            new_geometry = _run_decimation(per_part_target)
            actual_verts = sum(len(g.vertices) for g in new_geometry.values())
            if actual_verts == total_vertices:
                break
            # Scale face budget by the ratio of target to actual vertices.
            scale = total_vertices / actual_verts
            total_faces = max(1, int(round(total_faces * scale)))
            print(f"  Correction pass {iteration + 1}: actual={actual_verts}, "
                  f"adjusting to --total_faces {total_faces}")

        # Fine-tune: nudge face budget by ±1 until exact, stopping if we overshoot.
        if actual_verts != total_vertices:
            direction = 1 if actual_verts < total_vertices else -1
            for _ in range(20):
                total_faces += direction
                candidate = _run_decimation(_allocate_per_part(total_faces))
                candidate_verts = sum(len(g.vertices) for g in candidate.values())
                print(f"  Fine-tune: total_faces={total_faces} → {candidate_verts} vertices")
                if candidate_verts == total_vertices:
                    new_geometry = candidate
                    actual_verts = candidate_verts
                    break
                # Stop if we overshot past the target.
                if (direction == 1 and candidate_verts > total_vertices) or \
                   (direction == -1 and candidate_verts < total_vertices):
                    # Pick whichever is closer to the target.
                    if abs(candidate_verts - total_vertices) < abs(actual_verts - total_vertices):
                        new_geometry = candidate
                        actual_verts = candidate_verts
                    break
                new_geometry = candidate
                actual_verts = candidate_verts

        # Exact landing: if fine-tune still missed, collapse/split edges on the
        # largest part one at a time to hit the target vertex count exactly.
        if actual_verts != total_vertices:
            print(f"  Exact-landing: {actual_verts} → {total_vertices} via edge ops")
            new_geometry = _adjust_vertex_count(new_geometry, total_vertices)
            actual_verts = sum(len(g.vertices) for g in new_geometry.values())

    elif total_faces is not None:
        # If a global face budget was given, allocate per-part quotas in proportion
        # to each part's original face count (largest-remainder for rounding).
        if total_faces >= total_faces_before:
            print(f"  --total_faces ({total_faces}) >= current total "
                  f"({total_faces_before}); keeping parts as-is.")
        per_part_target = _allocate_per_part(total_faces)
        new_geometry = _run_decimation(per_part_target)
    else:
        # target_faces per-part or ratio mode
        new_geometry = {}
        for name, geom in scene.geometry.items():
            print(f"  Decimating part '{name}' ({len(geom.faces)} faces)")
            new_geometry[name] = decimate_trimesh(
                geom, target_faces=target_faces, ratio=ratio
            )

    # Replace geometries while preserving the scene graph.
    scene.geometry.clear()
    for name, geom in new_geometry.items():
        scene.geometry[name] = geom

    if total_vertices is not None:
        total_verts_after = sum(len(g.vertices) for g in scene.geometry.values())
        print(f"  Total vertices after: {total_verts_after}")
    else:
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
    parser.add_argument("--total_vertices", type=int, default=None,
                        help="Total vertex budget across all parts (proportional, converted to faces via V/F ratio)")

    args = parser.parse_args()

    specified = sum(x is not None for x in
                    (args.target_faces, args.ratio, args.total_faces, args.total_vertices))
    if specified == 0:
        parser.error("Specify one of --target_faces, --ratio, --total_faces, or --total_vertices")
    if specified > 1:
        parser.error("Specify only one of --target_faces, --ratio, --total_faces, --total_vertices")

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
                               total_faces=args.total_faces,
                               total_vertices=args.total_vertices)
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
                       total_faces=args.total_faces,
                       total_vertices=args.total_vertices)
