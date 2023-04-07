# SDC Final competition - Tracking

## Folder structure

:warning: 只保留我上傳的檔案和資料夾，其他所需的nuscenes資料和其他測試的檔案沒有放進去！

```bash!
nusc_tracking
|--- data
|    |--- CTrearrange_results
|    |    |--- eval
|    |    |--- centertrack_rearrange.json
|    |    |--- rearrangement.sh
|    |--- PTNN_track_results
|    |    |--- eval
|    |    |--- tracking_result.json
|    |    |--- track_template.sh
|    |--- NN_FusionTrack_results
|    |    |--- eval
|    |    |--- tracking_result.json
|    |    |--- track_fusion.sh
|    |--- LeakyReLU.pth
|    |--- centertrack_origin.json
|    |--- detection_results.json (provided by TA)
|    |--- frames_meta.json       (provided by TA)
|--- tools
|    |--- learning_score_update_function
|    |    |--- lsuf_network_module.py
|    |    |--- lsuf_train.py
|    |--- rearrangement.sh
|    |--- track_template.sh
|    |--- track_fusion.sh
|    |--- centerTrack_id_rearrangement.py
|    |--- track.py
|    |--- tracker.py
|    |--- track_fusion.py
|    |--- fusion.py
|--- README.md
```

## Final Result

- Method 1 result - AMOTA = 0.707

    :open_file_folder: &ensp; *data/PTNN_track_results/tracking_result.json*

- Method 2 result - AMOTA = 0.724 ( result file can be reached at two position )

    :open_file_folder: &ensp; *data/NN_FusionTrack_results/tracking_result.json*

    :open_file_folder: &ensp; *tracking_result.json*

## Method 1 : Track by lidar detection

> Based on center point method

1. open **tools/track_template.sh** and modify parameters inside

    ```bash=
    ## track_template.sh

    # Detection
    bbox_score=0.0

    # Tracker
    tracker='PointTracker' # choose : 'PointTracker', 'KF'
    min_hits=1
    max_age=6
    det_th=0.0
    del_th=0.0
    active_th=1.0
    score_decay=0.074 # tunning parameter
    score_update="multiplication" # choose : "nn", "multiplication", "none"
    use_vel=0
    model_path="data/LeakyReLU.pth" # for score_update = "nn"
    ......
    ```

2. run and get final result

    ```bash!
    bash tools/track_template.sh
    ```

3. Final result in **data/PTNN_track_results** folder

## Method 2 : Track by two sensor ( fusion )

> Based on **cneter point (lidar)** and **center track (camera)** method

1. Follow [CenterTrack](https://github.com/xingyizhou/CenterTrack) package installation and download pretrained model

2. Run nuScenes part ( command below ), then you will get the file of ***nuScenes_3Dtracking/results_nuscenes_tracking.json***, which includes the tracking result and detection score for each frame.

    ```bash!
    bash experiments/nuScenes_3Dtracking.sh
    ```

    :warning: **Rename json file to "centertrack_origin.json" and put into "nusc_tracking/data" (repository of Final competition)**

3. Back to **nusc_tracking** folder. Run the centerTrack_id_rearrangement.py

    ```bash!
    bash tools/rearrangement.sh
    ```

4. After get the ***centertrack_rearrange.json***, run **tools/track_fusion.sh** to get final results in **data/NN_FusionTrack_results**

    ```bash!
    bash tools/track_fusion.sh
    ```

5. Final result in **data/NN_FusionTrack_results** folder
