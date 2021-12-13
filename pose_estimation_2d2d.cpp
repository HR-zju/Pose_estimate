#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

// #include "extra.h" // use this if in OpenCV2

using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void gaussin(Eigen::Matrix<double, 8, 8> _UV,
             Eigen::Matrix<double, 8, 1> _E9,
             Eigen::Matrix<double, 8, 1> &_F);

void gramer(Eigen::Matrix<double, 8, 8> _UV,
            Eigen::Matrix<double, 8, 1> _E9,
            Eigen::Matrix<double, 8, 1> &_F);

void Jacobi(Eigen::Matrix<double, 8, 8> _UV,
            Eigen::Matrix<double, 8, 1> _E9,
            Eigen::Matrix<double, 8, 1> &_F);

void Gauss_Seidel(Eigen::Matrix<double, 8, 8> _UV,
                  Eigen::Matrix<double, 8, 1> _E9,
                  Eigen::Matrix<double, 8, 1> &_F);

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);  // matches
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    //// reference
    //-- 估计两张图像间运动
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //-- 验证E=t^R*scale
    Mat t_x =
            (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R=" << endl << t_x * R << endl;

    //-- 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }


    ///// my method
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> UV;
    UV.resize(matches.size(), 8);
    for (int keypoints_idx = 0; keypoints_idx < matches.size(); ++keypoints_idx) {
        UV(keypoints_idx, 0) = keypoints_2[keypoints_idx].pt.x * keypoints_1[keypoints_idx].pt.x;
        UV(keypoints_idx, 1) = keypoints_2[keypoints_idx].pt.x * keypoints_1[keypoints_idx].pt.y;
        UV(keypoints_idx, 2) = keypoints_2[keypoints_idx].pt.x;
        UV(keypoints_idx, 3) = keypoints_2[keypoints_idx].pt.y * keypoints_1[keypoints_idx].pt.x;
        UV(keypoints_idx, 4) = keypoints_2[keypoints_idx].pt.y * keypoints_1[keypoints_idx].pt.y;
        UV(keypoints_idx, 5) = keypoints_2[keypoints_idx].pt.y;
        UV(keypoints_idx, 6) = keypoints_1[keypoints_idx].pt.x;
        UV(keypoints_idx, 7) = keypoints_1[keypoints_idx].pt.y;
    }

    Eigen::Matrix<double, 8, 8> UVT;
    UVT = UV.transpose() * UV;

    Eigen::Matrix<double, 8, 1> F;
    Eigen::Matrix<double, 79, 1> Zero = Eigen::Matrix<double, 79, 1>::Ones();
    double e9 = -1;
    Zero = Zero * e9;
    Eigen::Matrix<double, 8, 1> Zero2;
    Zero2 = UV.transpose() * Zero;

    /// prepare data
    vector<Point2f> points1;
    vector<Point2f> points2;
    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    cv::Mat cvF;  // 基础矩阵
    cv::Mat M_R;
    cv::Mat M_t;  // SVD 分解得到的旋转矩阵和平移矩阵
    clock_t start_time;
    clock_t end_time;
    Eigen::Matrix3d Fundmental;

    /// Qr
    start_time = clock();
    F = UVT.colPivHouseholderQr().solve(Zero2);
    end_time = clock();
    Fundmental << F, 1;
    cv::eigen2cv(Fundmental, cvF);
    recoverPose(cvF, points1, points2, M_R, M_t, focal_length, principal_point);

    cout << "====QR_method====" << endl;
    cout << "Time cost: " << 1000 * (end_time - start_time) / (double) CLOCKS_PER_SEC << " ms" << endl;
    cout << "F: " << F.transpose() << endl;
    cout << "R: " << endl << M_R << endl;
    cout << "t: " << endl << M_t << endl;

    /// cholesky
    start_time = clock();
    F = UVT.ldlt().solve(Zero2);
    end_time = clock();
//    Eigen::Matrix3d Fundmental;
    Fundmental << F, 1;
    cv::eigen2cv(Fundmental, cvF);
    recoverPose(cvF, points1, points2, M_R, M_t, focal_length, principal_point);

    cout << "====cholesky_method====" << endl;
    cout << "Time cost: " << 1000 * (end_time - start_time) / (double) CLOCKS_PER_SEC << " ms" << endl;
    cout << "F: " << F.transpose() << endl;
    cout << "R: " << endl << M_R << endl;
    cout << "t: " << endl << M_t << endl;



    /// 高斯消去法则
    start_time = clock();
    gaussin(UVT, Zero2, F);
    end_time = clock();
//    Eigen::Matrix3d Fundmental;
    Fundmental << F, 1;
    cv::eigen2cv(Fundmental, cvF);
    recoverPose(cvF, points1, points2, M_R, M_t, focal_length, principal_point);

    cout << "====gauss_method====" << endl;
    cout << "Time cost: " << 1000 * (end_time - start_time) / (double) CLOCKS_PER_SEC << " ms" << endl;
    cout << "F: " << F.transpose() << endl;
    cout << "R: " << endl << M_R << endl;
    cout << "t: " << endl << M_t << endl;

    /// Gram法则
    start_time = clock();
    gramer(UVT, Zero2, F);
    end_time = clock();
//    Eigen::Matrix3d Fundmental;
    Fundmental << F, 1;
    cv::eigen2cv(Fundmental, cvF);
    recoverPose(cvF, points1, points2, M_R, M_t, focal_length, principal_point);

    cout << "====Gramer_method====" << endl;
    cout << "Time cost: " << 1000 * (end_time - start_time) / (double) CLOCKS_PER_SEC << " ms" << endl;
    cout << "F: " << F.transpose() << endl;
    cout << "R: " << endl << M_R << endl;
    cout << "t: " << endl << M_t << endl;

    /// Jacobi法则
    start_time = clock();
    Jacobi(UVT, Zero2, F);
    end_time = clock();

    Fundmental << F, 1;
    cv::eigen2cv(Fundmental, cvF);
    recoverPose(cvF, points1, points2, M_R, M_t, focal_length, principal_point);

    cout << "====Jacobi_method====" << endl;
    cout << "Time cost: " << 1000 * (end_time - start_time) / (double) CLOCKS_PER_SEC << " ms" << endl;
    cout << "F: " << F.transpose() << endl;
    cout << "R: " << endl << M_R << endl;
    cout << "t: " << endl << M_t << endl;


    /// Gauss-Seidel法则
    start_time = clock();
    Gauss_Seidel(UVT, Zero2, F);
    end_time = clock();

    Fundmental << F, 1;
    cv::eigen2cv(Fundmental, cvF);
    recoverPose(cvF, points1, points2, M_R, M_t, focal_length, principal_point);

    cout << "====Gauss-Seidel====" << endl;
    cout << "Time cost: " << 1000 * (end_time - start_time) / (double) CLOCKS_PER_SEC << " ms" << endl;
    cout << "F: " << F.transpose() << endl;
    cout << "R: " << endl << M_R << endl;
    cout << "t: " << endl << M_t << endl;

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {
    // 相机内参,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  // already konw

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //-- 计算单应矩阵
    //-- 但是本例中场景不是平面，单应矩阵意义不大
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    // 此函数仅在Opencv3中提供
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

}


void gaussin(Eigen::Matrix<double, 8, 8> _UV, Eigen::Matrix<double, 8, 1> _E9, Eigen::Matrix<double, 8, 1> &_F) {
    int n = _UV.cols();

    //判断能否用高斯消元法，如果矩阵主对角线上有0元素存在是不能用的
    for (int i = 0; i < n; i++)
        if (_UV(i, i) == 0) {
            cout << "can't use gaussin meathod" << endl;
            return;
        }

    int i, j, k;
    double c[n];    //存储初等行变换的系数，用于行的相减
    //消元的整个过程如下，总共n-1次消元过程。
    for (k = 0; k < n - 1; k++) {
        //求出第K次初等行变换的系数
        for (i = k + 1; i < n; i++)
            c[i] = _UV(i, k) / _UV(k, k);

        //第K次的消元计算
        for (i = k + 1; i < n; i++) {
            for (j = 0; j < n; j++) {
                _UV(i, j) = _UV(i, j) - c[i] * _UV(k, j);
            }
            _E9[i] = _E9[i] - c[i] * _E9[k];
        }
    }

    // 解的存储数组
    double x[n];
    // 先计算出最后一个未知数；
    x[n - 1] = _E9[n - 1] / _UV(n - 1, n - 1);
    // 求出每个未知数的值
    for (i = n - 2; i >= 0; i--) {
        double sum = 0;
        for (j = i + 1; j < n; j++) {
            sum += _UV(i, j) * x[j];
        }
        x[i] = (_E9[i] - sum) / _UV(i, i);
    }

    cout << " the solution of the equations is:" << endl;
    cout << endl;
    for (i = 0; i < n; i++)
        cout << "x" << i + 1 << "=" << x[i] << endl;

    _F << x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7];

}

void gramer(Eigen::Matrix<double, 8, 8> _UV, Eigen::Matrix<double, 8, 1> _E9, Eigen::Matrix<double, 8, 1> &_F) {
    Eigen::Matrix<double, 8, 8> UV_star = _UV;
    for (int i = 0; i < 8; ++i) {
        UV_star = _UV;
        UV_star.col(i) = _E9;
        _F(0, i) = UV_star.determinant() / _UV.determinant();
    }
}

void Jacobi(Eigen::Matrix<double, 8, 8> _UV, Eigen::Matrix<double, 8, 1> _E9, Eigen::Matrix<double, 8, 1> &_F) {
    // 检查主元是否都合理
    int n = _UV.cols();
    for (int i = 0; i < n; i++) {
        if (_UV(i, i) == 0) {
            cout << "can't use gaussin meathod" << endl;
            return;
        }
    }
    Eigen::Matrix<double, 8, 8> Bj;
    Eigen::Matrix<double, 8, 1> fj, pre_x, curr_x;

    cout << "_UV: " << endl << _UV << endl;

    for (int fj_idx = 0; fj_idx < fj.rows(); ++fj_idx) {
        fj(fj_idx, 0) = _E9(fj_idx, 0) / _UV(fj_idx, fj_idx);
    }
    cout << "fj: " << endl << fj << endl;

    Eigen::Matrix<double, 8, 8> dia_UV = Eigen::Matrix<double, 8, 8>::Zero();
    for (int ii = 0; ii < dia_UV.rows(); ++ii) {
        dia_UV(ii, ii) = 1.0 / _UV(ii, ii);
    }

    cout << "Bj " << endl << Bj.eigenvalues().real().maxCoeff() << endl;
    Bj = (-dia_UV) * _UV + Eigen::Matrix<double, 8, 8>::Identity();
    // 判断是否收敛
    Eigen::VectorXd::Index maxRow, maxCol;
    Eigen::VectorXd::Index minRow, minCol;
    if (abs(Bj.eigenvalues().real().minCoeff()) > 1.0 || abs(Bj.eigenvalues().real().maxCoeff()) > 1.0) {
        cout << "B最大特征值为： " << Bj.eigenvalues().real().minCoeff() << endl << "迭代不收敛" << endl;
        return;
    }

    pre_x = Eigen::Matrix<double, 8, 1>::Zero();  // 初始为0向量
    curr_x = Bj * pre_x + fj;
    while ((curr_x - pre_x).lpNorm<Eigen::Infinity>() / curr_x.lpNorm<Eigen::Infinity>() >= 1e-2) {
        pre_x = curr_x;
        curr_x = Bj * pre_x + fj;
    }
    _F = curr_x;
}

void Gauss_Seidel(Eigen::Matrix<double, 8, 8> _UV,
                  Eigen::Matrix<double, 8, 1> _E9,
                  Eigen::Matrix<double, 8, 1> &_F) {                                          //定义Gauss-Seidel迭代函数

    int n = 8;
    double a[8][8], b[8], x[8][8];
    double e = 1e-2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)  //输入系数矩阵
            a[i][j] = _UV(i, j);  //输入等式右边的矩阵
        b[i] = _E9(i, 0);
    }

    for (int k = 1; k <= 100; k++) {
        for (int i = 0; i < n; i++) {
            double s1 = 0, s2 = 0, s = 0;
            x[k][i] = 1.0 / a[i][i];
            double re = b[i];  //每一行遍历完re重置
            for (int j = 0; j < n; j++) {
                if (i < j)   //上三角部分乘系数求和
                    s1 += a[i][j] * x[k - 1][j];
                else if (i > j)  //下三角部分乘系数求和
                    s2 += a[i][j] * x[k][j];
            }
            s = s1 + s2;  //每一行遍历完求和
            re -= s;
            x[k][i] *= re;
        }

        bool judge = true;
        for (int i = 0; i < n; i++)
            if (fabs(x[k - 1][i] - x[k][i]) > e) {
                judge = false;
                break;
            }
        if (judge == true) {
            _F << x[k][0], x[k][1], x[k][2],
                    x[k][3], x[k][4], x[k][5],
                    x[k][6], x[k][7], x[k][8];
            return;
        }
        else {
            cout << "Gauss_Seidel不收敛" << endl;
        }
    }
}
