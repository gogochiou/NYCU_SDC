# Detection
bbox_score=0.0

# Tracker
tracker='PointTracker'
# tracker='KF'
min_hits=1
max_age=6
det_th=0.0
del_th=0.0
active_th=1.0
score_decay=0.075
# score_update="multiplication"
score_update="nn"
use_vel=0
model_path="data/LeakyReLU.pth"

# Fusion
# Tracking2="data/KF_track_results/tracking_result.json"
Tracking2="data/CTrearrange_results/centertrack_rearrange.json"
decay1=0.2
decay2=0.4
star=True
fusion_del_th=0.0
v_min=0.4
v_max=1.9
v_weight=0.8

# Data and eval
evaluate=1
dataroot="data/nuscenes"
split="val"
out_dir="data/NN_FusionTrack_results"
detection_path="data/detection_result.json"
frames_meta_path="data/frames_meta.json"

python tools/track_fusion.py --split $split --out_dir $out_dir \
--min_hits $min_hits --det_th $det_th --del_th $del_th --active_th $active_th \
--score_decay $score_decay --score_update $score_update \
--Tracking2 $Tracking2 --decay1 $decay1 --decay2 $decay2 --star $star \
--fusion_del_th $fusion_del_th --v_min $v_min --v_max $v_max --v_weight $v_weight \
--evaluate $evaluate --dataroot $dataroot \
--detection_path $detection_path --frames_meta_path $frames_meta_path \
--tracker $tracker --bbox-score $bbox_score --max_age $max_age \
--use_vel $use_vel --model_path $model_path

# Copy arguments to out_dir
cp tools/track_fusion.sh $out_dir/track_fusion.sh