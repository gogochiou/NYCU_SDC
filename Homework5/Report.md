# **LiTAMIN2 : Ultra Light LiDAR-based SLAM**

> Student ID : A111137
>
> Name : 邱柏鎧

## **Introduction**

> Ultra Light LiDAR-based SLAM using Geometric Approximation applied with KL-Divergence

### **Abstract :**

In this paper, a 3D light detection and ranging simultaneous localization and mapping (SLAM) method is proposed. It is available for operating upon 500~1000Hz with high accuracy (almost same as the state-of-the-art method - "SuMa"), for more precise result, it can still work on 200Hz.

<center><img src=./Picture/compare.png width=85%></center>

This paper uses a novel ICP metric to speed up the registration process while maintaining accuracy. However, Reducing quantity of point cloud can drop the accuracy, to avoid this issue, symmetric KL-divergence is introduced to the ICP cost that reflects the difference between two probabilistic. The cost function includes not only the distance between points but also differences between distribution shapes.

### **Main works**

- Reduction of the number of points

    1. Voted a group of input points into the voxel grids.

    2. Aligned them using the means of the voting points.

    3. Integrated the point clouds into voxel map.

    ![reducing PCL](./Picture/reducingPCL.png)

- ICP cost function applied with symmetric KL-divergence

    ![ICP cost](./Picture/ICP_cost.png)

## **Reference :**

### **Video**

- [LiTAMIN2 (ICRA 2021) Youtube](https://www.youtube.com/watch?v=cDpMtXU6gQU)

![video_clip](./Picture/video_clip.png)

### **Links**

- [MR2T lab official web](https://unit.aist.go.jp/hcmrc/mr-rt/project.html)

- [LiTAMIN2: Ultra Light LiDAR-based SLAM using Geometric Approximation applied with KL-Divergence](https://arxiv.org/abs/2103.00784)

- [LiTAMIN2 introduction](https://zhuanlan.zhihu.com/p/378147891)

- [ICRA2021 LiTAMIN2的復現](https://blog.csdn.net/a850565178/article/details/122398679)