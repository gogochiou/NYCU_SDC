# Detection
bbox_score=0.0

# Tracker
# tracker='PointTracker'
tracker='KF'
min_hits=1
max_age=6
det_th=0.0
del_th=0.0 #0.01
active_th=1.0
score_decay=0.2
score_update="multiplication"
use_vel=0

# Data and eval
evaluate=1
dataroot="data/nuscenes"
split="val"
out_dir="data/KF_track_results"
detection_path="data/detection_result.json"
frames_meta_path="data/frames_meta.json"

python tools/track.py --split $split --out_dir $out_dir \
--min_hits $min_hits --det_th $det_th --del_th $del_th --active_th $active_th \
--score_decay $score_decay --score_update $score_update \
--evaluate $evaluate --dataroot $dataroot \
--detection_path $detection_path --frames_meta_path $frames_meta_path \
--tracker $tracker --bbox-score $bbox_score --max_age $max_age \
--use_vel $use_vel

# Copy arguments to out_dir
cp tools/track_template.sh $out_dir/track_template.sh