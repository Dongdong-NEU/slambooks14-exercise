//
// Created by hidongxi on 2023/2/21.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

using namespace std;

string image_file = "/home/xhd/视频/my_vslam/slambooks14-exercise/image/test.png";   // 请确保路径正确

int main(int argc, char **argv) {


    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    cv::Mat image = cv::imread(image_file,0);   // 图像是灰度图，CV_8UC1
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图
    // 获取起始时间点
    auto start = std::chrono::high_resolution_clock::now(); 
    // 计算去畸变后图像的内容
    for(int i = 0; i < 1; i++){
    for (int v = 0; v < rows; v++){
        for (int u = 0; u < cols; u++) {
            double u_distorted = 0, v_distorted = 0;
            // start your code here
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);

            u_distorted = x * (1 + k1*r*r + k2*r*r*r*r) + 2 * p1 * x * y + p2*(r*r + 2*x*x);
            v_distorted = y * (1 + k1*r*r + k2*r*r*r*r) + 2 * p2 * x * y + p1*(r*r + 2*y*y);

            u_distorted = fx * u_distorted + cx;
            v_distorted = fy * v_distorted + cy;
            // end your code here

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    } 
    }
    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now(); 
    // 计算时间差，以微秒为单位
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); 
    std::cout << "程序运行时间为 " << duration.count() / 1000 << " 毫秒" << std::endl;
    // 画图去畸变后图像
    cv::imshow("my image undistorted ", image_undistort);
    cv::waitKey(0);

    auto start2 = std::chrono::high_resolution_clock::now();
    //OpenCV去畸变函数的使用
    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 458.645, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> ( 4,1 ) << -0.28340811,  0.07395907, 0.00019359, 1.76187114e-05 );

    // const string str = "/home/jiang/4_learn/WeChatCode/ImageUndistort/data/";
    const int nImage = 1;
    const int ImgWidth = image.cols;
    const int ImgHeight = image.rows;

    cv::Mat map1, map2;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 5;
    
    //                                               内参, 畸变系数, 图像尺寸,比例因子,输出图像尺寸, 输出感兴趣区域设置;
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);

    cv::initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);

    for(int i = 0; i < nImage; i++)
    {
        // string InputPath = str + to_string(i) + ".png";
        // cv::Mat RawImage = cv::imread(InputPath);
        // cv::imshow("RawImage", RawImage);
        cv::Mat UndistortImage;
        //           原图,  矫正后图,        内参,畸变参数,新相机内参;
        // cv::undistort(image, UndistortImage, K, D, K);

        // 注意:如果undistort函数中第五个参数放getOptimalNewCameraMatrix,alpha=1(全保留)的新相机内参的话,所有像素将会被保留.
        // cv::undistort(image, UndistortImage, K, D, K);

        cv::remap(image, UndistortImage, map1, map2, cv::INTER_LINEAR);
        cv::imshow("UndistortImage", UndistortImage);
        cv::waitKey(0);
        // string OutputPath = str + to_string(i) + "_un2" + ".png";
        // cv::imwrite(OutputPath, UndistortImage);       
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2); // 计算时间差，以微秒为单位
    std::cout << "程序运行时间为 " << duration2.count() / 1000 << " 毫秒" << std::endl;

    return 0;
}
