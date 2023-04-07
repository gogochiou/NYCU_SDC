# Report

## DownSampling

- [【PCL笔记】Filtering 点云滤波](https://zhuanlan.zhihu.com/p/95983353)

- [使用voxel稀疏點雲](https://blog.csdn.net/u011021773/article/details/78941001)

## ICP

- [ICP 學習筆記](https://blog.csdn.net/u010696366/article/details/8941938)

- [點雲精配准](https://zhuanlan.zhihu.com/p/107218828)

- [PCL-ICP(IterativeClosestPoint)源码解析](https://blog.csdn.net/zack_liu/article/details/117991984)

```cpp
pcl::Registration<PointSource, PointTarget, Scalar>::align (PointCloudSource &output, const Matrix4& guess)
```

```cpp
pcl::Registration<PointSource, PointTarget, Scalar>::getFitnessScore (double max_range)
// while calculating the score, we only consider the error distance beneath 'max_range'
// error distance represent distance between transformed source and target 
```

## PCL

- [两种点云地面去除方法](https://blog.csdn.net/qq_38167930/article/details/119165988)

- [两种读取和写入pcd点云文件的方法](https://blog.csdn.net/liukunrs/article/details/80769145)

## Eigen

- [Eigen The Matrix class](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html)