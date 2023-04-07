#!/usr/bin/env python3

from re import X
from tkinter import Scale
import rospy
import math, os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from itertools import chain
# KF coded by yourself
from kalman_filter6DOF import KalmanFilter6DOF

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.KF = None
        self.step = 0   # Record update times

        self.last_odometry_position = np.zeros(2)
        self.last_odometry_angle = 0
        self.last_diff_cov = np.ones((3,3))
        self.last_diff = np.zeros(2)
        self.last_diff_yaw = 0

        self.gt_list = []
        self.est_list = []

    def shutdown(self):
        print("shuting down fusion.py")

    def gpsCallback(self, data):
        self.step += 1
        # Get GPS data only for 2D (x, y)
        measurement = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)[:3,:3]
        gps_covariance[2,2] = 100000
        # print(f"GPS cov : {gps_covariance}")    

        # KF update
        if self.step == 1:
            self.init_KF(measurement[0], measurement[1], 0, 0, 0, 0)
        else:
            self.KF.Q = gps_covariance
            self.KF.update(z = measurement)
        print(f"P_update: {self.KF.P}")
        print(f"estimation_update: {self.KF.x}")
        print("------------------------")

    def odometryCallback(self, data):
        self.step += 1
        # Read radar odometry data from ros msg
        position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        odometry_covariance = np.identity(6)*6
        # odometry_covariance = np.zeros((6,6))
        # because cov of yaw will be 0 for input, so we need to eliminate it for avoiding 0/1 become nan
        odometry_covariance[:2, :2] = np.array(data.pose.covariance).reshape(6, -1)[:2,:2]/self.last_diff_cov[:2,:2]
        odometry_covariance[3:6, 3:6] = np.abs(np.array(data.pose.covariance).reshape(6, -1)[:3,:3])
        # odometry_covariance[:3, :3] = np.abs(np.array(data.pose.covariance).reshape(6, -1)[:3,:3])
        # odometry_covariance[3:5, 3:5] = np.array(data.pose.covariance).reshape(6, -1)[:2,:2]/self.last_diff_cov[:2,:2]

        odometry_covariance[:3, 3:6] = np.identity(3)
        odometry_covariance[3:6, :3] = np.identity(3)
        
        print(f"Radar cov : \r\n{odometry_covariance}\r\n") 

        # Get euler angle from quaternion
        orientation = data.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, 
                                                  orientation.z, orientation.w])

        # Calculate odometry difference
        diff = position - self.last_odometry_position
        diff_yaw = yaw - self.last_odometry_angle
        ddiff = diff - self.last_diff
        ddiff_yaw = diff_yaw - self.last_diff_yaw

        # KF predict
        if self.step == 1:
            self.init_KF(position[0], position[1], yaw, 0, 0, 0)
        else:
            # self.KF.R = (odometry_covariance*1000000)
            self.KF.R = odometry_covariance
            self.KF.predict(u = np.array([position[0], position[1], yaw, diff[0], diff[1], diff_yaw]))
            # self.KF.predict(u = np.array([diff[0], diff[1], diff_yaw, ddiff[0], ddiff[1], ddiff_yaw]))
        print(f"estimation_predict: \r\n{self.KF.x}\r\n")
        print(f"P_predict: \r\n{self.KF.P}\r\n")
        self.last_odometry_position = position
        self.last_odometry_angle = yaw
        row, col = np.diag_indices_from(self.last_diff_cov)
        self.last_diff_cov[row, col] = np.diagonal(np.array(data.pose.covariance).reshape(6, -1)[:3,:3])
        self.last_diff_cov[2, 2] = 1
        self.last_diff = diff
        self.last_diff_yaw = diff_yaw

        quaternion = quaternion_from_euler(0, 0, self.KF.x[2])

        # Publish odometry with covariance
        # list(chain.from_iterable(self.KF.P.tolist()))
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        predPose.pose.pose.position.x = self.KF.x[0]
        predPose.pose.pose.position.y = self.KF.x[1]
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        predPose.pose.covariance = [self.KF.P[0][0], self.KF.P[0][1],0,0,0,0,
                                    self.KF.P[1][0], self.KF.P[1][1],0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0,
                                    0,0,0,0,0,0 ]
        self.posePub.publish(predPose)

    def gtCallback(self, data):
        if self.step >=1 :
            gt_position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
            kf_position = self.KF.x[:2]
            self.gt_list.append(gt_position)
            self.est_list.append(kf_position)

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        gt_x, gt_y = zip(*self.gt_list)
        est_x, est_y = zip(*self.est_list)
        plt.plot(gt_x, gt_y, alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(est_x, est_y, alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("/home/ee904/SDC/hw4/src/kalman_filter/results"):
            os.mkdir("/home/ee904/SDC/hw4/src/kalman_filter/results")
        plt.savefig("/home/ee904/SDC/hw4/src/kalman_filter/results/result.png")
        plt.show()

    def init_KF(self, x, y, yaw, dx, dy, dyaw):
        # Initialize the Kalman filter when the first data comes in
        self.KF = KalmanFilter6DOF(x = x, y = y, yaw = yaw, dx = dx, dy = dy, dyaw = dyaw)
        # self.KF.A = np.identity(3)
        # self.KF.B = np.identity(3)

if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    # fusion.plot_path()
