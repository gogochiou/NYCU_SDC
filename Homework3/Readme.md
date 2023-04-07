# **SDC Homework 3 - Kalman Filter**

## **Part 1 : warm-up**

### **Environment Setup**

In the following homeworks , we would use container to finish courseworks to prevent package dependancy or environment version issues. Please make sure to have some basic knowledge of Docker.

#### **Prerequisite**

- Docker version >= 19.03
- Make sure **~/catkin_ws/src** existing
- Place **sdc_hw3/** into **~/catkin_ws/sdc_hw3**

#### **Using pre-built images**

We wrap ROS noetic, Python 3.8 and the package we use this time e.g. NumPy, imageio into pre-built images. Please follow two commands to enter the environment.

- Main Terminal

```bash
xhost +local:
```

```bash
sudo docker run \
-it \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-p 2233:22 \
--rm \
--name ros \
--user root \
-e GRANT_SUDO=yes \
-v ~/catkin_ws:/root/catkin_ws \
softmac/sdc-course-docker:latest \
bash
```

- Enter container in other terminals

```bash
docker exec -it ros bash 
```

## **Part 2 : run the code**

### **Objective**

In this homework, you will need to understand and implement Kalman filter. Notice that you can only use Numpy and the python standard libraries, other packages(ex: filterpy) are not allowed.

### **Requirements**

1. Implement Kalman filter with robot state [x, y, yaw].
2. Plot the outputs of Kalman filter and measurements.

### **Implementation Details**

- Description

    Inputs generation and plotting functions are provided in the sample code. You will only need to complete the algorithm in kalman_filter.py.

    ![KF_Pseudo Code](./Picture/KF_pseudoCode.png)

    Inputs are given as:

    Control term (u): displacement of robot and yaw change [delta_x, delta_y, delta_yaw] (with 0 mean, 0.1 variance Gaussian noise added to delta_x, delta_y, and delta_yaw, respectively)

    Measurement (z): Position of [x, y] (with 0 mean, 0.75 variance added to x and y, respectively).

- Test the algorithm (you need to enter sdc_hw3 folder and in docker mode)

    ```bash
    python3 Kalman2D.py
    ```