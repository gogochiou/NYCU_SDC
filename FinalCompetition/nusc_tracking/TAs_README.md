# Final competition - Tracking
This task is based on tracking by detection method. We already have detection results (Bounding boxes) from detector. Our goal is to develope a good tracking algorithm.
### Date: 12/8 ~ 12/29

## Data preparation
- **Detection result** and **frames meta data** from [**google drive**](https://drive.google.com/drive/folders/13jmwcS2qu89QftSmrWmGpgQu20gF8YPl?usp=share_link).
- If you want to **evaluate** and **visualize**, you will need to download **nuscenes trainval dataset** from [**Nuscenes website**](https://www.nuscenes.org/nuscenes#download). (If you don't use camera, you don't need to download camera data)
    - Size: about 320GB (with camera)
    - Size: about 160GB (without camera)
- The folder structure should be organized as follows before processing.
```
nusc_tracking
├── configs
├── docker
├── ros_ws
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-trainval
```

## Environment setup
**First** of all, cd to your workspace and,
```bash!
git clone https://github.com/derekray311511/nusc_tracking.git
```
Modify docker/run.sh **line1** and **line2**.
```bash=
your_data_path='/your/data/path'
your_workspace_path='/your/nusc_tracking'

xhost +local:

docker run \
-it \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--network host \
--rm \
--name tracking \
-e GRANT_SUDO=yes \
-v $your_data_path:/data \
-v $your_workspace_path:/home/Student/Tracking \
tracking \
bash
```
```bash!
cd nusc_tracking 
# Build the docker imagecd nusc_tracking 
cd docker && docker build . -t tracking
# Run the docker image and create a container
bash run.sh
```
If you want to run the same container in other terminal:
```bash!
docker exec -it tracking bash
```
Create a virtual path to data folder in docker
```bash!
cd /home/Student/Tracking
ln -fsv /data data
```
Put the `detection_result.josn` and `frames_meta.json` (download from PART1) into your `data` folder

## Tracking and Evaluation
If you want to evaluate the tracking result, please set parameter **evaluate** in tools/track_template.sh to **1** (True).
```bash!
# Arguments can be set in track_template.sh
bash tools/track_template.sh
```

## Visualization
Need to download dataset from `nuscenes trainval dataset` from [**Nuscenes website**](https://www.nuscenes.org/nuscenes#download).
We use **ros** to **visualize** our tracking results.
```bash!
cd ros_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
# First terminal
roscore
# Second terminal
rviz -d ../configs/track.rviz # can run out of the docker container
# Third terminal
bash src/visualize.sh
```
If you don't have camera data, please comment these two lines in `visualize.py`
![](https://i.imgur.com/jxTrD8d.png)


## Hand in the results
GOOGLE DRIVE with a eval result excel file