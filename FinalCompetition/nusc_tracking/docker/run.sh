your_data_path='/home/gogochiou/Gogochiou_Workspace/NTHU_course/NYCU_SDC/FinalCompetition/nusc_tracking/data'
your_workspace_path='/home/gogochiou/Gogochiou_Workspace/NTHU_course/NYCU_SDC/FinalCompetition/nusc_tracking'

xhost +local:

sudo docker run \
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
