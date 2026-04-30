# 1. Remesh the ply files and convert to glb format
for ply in processed_object_meshes/*ply; do
  ply_name=$(basename $ply)
  obj_name=${ply_name%.*}
  python remesh.py --input $ply --output data/${obj_name}/${obj_name}.glb
done

# 2. Run partfield inference
for glb_path in arctic/data/*; do
  obj_name=$(basename $glb_path)
  python partfield_inference.py \
    -c configs/final/demo.yaml \
    --opts continue_ckpt \
    model/model_objaverse.ckpt \
    result_name partfield_features/${obj_name} \
    dataset.data_path ${glb_path}
done

# 3. Run clustering
for glb_path in arctic/data/*; do
# for glb_path in $(ls -rd data/*); do
  obj_name=$(basename $glb_path)
  python run_part_clustering.py \
    --root exp_results/partfield_features/${obj_name} \
    --dump_dir exp_results/clustering/${obj_name} \
    --source_dir ${glb_path} \
    --use_agglo True \
    --max_num_clusters 10 \
    --option 1 \
    --with_knn True \
    --normal_threshold 0.1 \
    --refine_iters 5 \
    --ring_size 10
done

# 4. Copy the clustering results (ply files) to clustering_final based on the number of parts
mkdir -p exp_results/clustering_final
while read obj_name k; do
  printf -v k_pad "%02d" "$k"
  cp exp_results/clustering/${obj_name}/ply/${obj_name}_0_${k_pad}.ply exp_results/clustering_final/${obj_name}_${k_pad}.ply
done < exp_results/num_part.txt

# 5. Convert to usdz format with separated parts
python mesh_to_usdz.py --input_dir exp_results/clustering_final/ --output_dir exp_results/usdz/

# 6. Downsample
for obj in arctic/data_with_parts/*; do
  obj_name=$(basename $obj)
  python downsample_mesh.py \
  --input $obj \
  --output arctic/data_256/${obj_name} \
  --total_vertices 256
done

for obj in arctic/data_with_parts/*; do
  obj_name=$(basename $obj)
  python downsample_mesh.py \
  --input $obj \
  --output arctic/data_128/${obj_name} \
  --total_vertices 128
done


conda activate text2hoi

# 1. Find contact parts
python find_contact_parts.py --contact --output out.json --format json

python find_contact_parts.py --dataset arctic --contact \
  --data_path data/arctic/data.npz \
  --obj_pkl   data/arctic/obj.pkl \
  --glb_dir   data/arctic/data_with_parts \
  --output    out_arctic.json --format json


# 2. Extend prompts with contact parts
python extend_prompts_with_parts.py \
  --text_json data/grab/text.json \
  --contacts_json out.json \
  --output data/grab/text_with_parts.json

python extend_prompts_with_parts.py \
  --data_npz data/arctic/data.npz \
  --text_json data/arctic/text.json \
  --contacts_json out_arctic.json \
  --output data/arctic/text_with_parts.json

python extend_prompts_with_parts.py --dataset arctic \
  --data_npz data/arctic/data.npz \
  --text_json data/arctic/text.json \
  --contacts_json out_arctic.json