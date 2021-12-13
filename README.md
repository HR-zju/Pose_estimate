# 科学与工程计算方法大作业

使用“科学与工程计算方法”课程上所学的第三章“线性代数方程组的数值解法”来估计相机的运动。

已知相机在同一场景下拍摄的两幅不同视角的图片。假设画面中的物体是静止的，相机的运动和物体像素坐标的移动是一一对应的。也就是说可以通过两幅画面的特征点匹配结果，还原出相机的相对运动。

### 编译

```bash
git clone https://github.com/HR-zju/Pose_estimate.git
mkdir build && cd build
cmake ..
make -j4
cd ..
build/pose_estimation_2d2d 1.png 2.png
```

即可在终端中看到文中的结果。
