# Detection
bbox_score=0.0

# Tracker
ct_result="data/centertrack_origin.json"
frames_meta_path="data/frames_meta.json"

# Data and eval
evaluate=1
dataroot="data/nuscenes"
split="val"
out_dir="data/CTrearrange_results"

python tools/centerTrack_id_rearrangement.py --split $split --out_dir $out_dir \
--evaluate $evaluate --dataroot $dataroot \
--ct_result $ct_result --frames_meta_path $frames_meta_path

# Copy arguments to out_dir
cp tools/rearrangement.sh $out_dir/rearrangement.sh