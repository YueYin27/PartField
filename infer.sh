# 1. Remesh the ply files and convert to glb format
for ply in data/processed_object_meshes/*.ply; do
  ply_name=$(basename $ply)
  obj_name=${ply_name%.*}
  python remesh.py --input $ply --output data/${obj_name}/${obj_name}.glb
done

# 2. Run partfield inference
for glb_path in data/*; do
  obj_name=$(basename $glb_path)
  python partfield_inference.py \
    -c configs/final/demo.yaml \
    --opts continue_ckpt \
    model/model_objaverse.ckpt \
    result_name partfield_features/${obj_name} \
    dataset.data_path ${glb_path}
done

# 3. Run clustering
for glb_path in data/*; do
# for glb_path in $(ls -rd data/*); do
  obj_name=$(basename $glb_path)
  python run_part_clustering.py \
    --root exp_results/partfield_features/${obj_name} \
    --dump_dir exp_results/clustering/${obj_name} \
    --source_dir ${glb_path} \
    --use_agglo True \
    --max_num_clusters 20 \
    --option 1 \
    --with_knn True \
    --normal_threshold 0.1 \
    --refine_iters 5 \
    --ring_size 10
done

# 4. Split the parts and save as ply files
while read obj_name k; do
  printf -v k_pad "%02d" "$k"
  python split_parts.py \
    --mesh exp_results/partfield_features/${obj_name}/input_${obj_name}_0.ply \
    --labels exp_results/clustering/${obj_name}/cluster_out/${obj_name}_0_${k_pad}.npy \
    --out_dir exp_results/parts/${obj_name}
done < exp_results/num_part.txt
