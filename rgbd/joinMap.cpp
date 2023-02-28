#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>


using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    TrajectoryType poses;         // 相机位姿

    ifstream fin("/home/xhd/视频/my_vslam/slambooks14-exercise/rgbd/pose.txt");
    if (!fin) {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }

    // for (int i = 0; i < 1; i++) 
    // {
        int i = 0;
        boost::format fmt("./%s/%d.%s"); //图像文件格式

        // cv::Mat image_color = cv::imread((fmt % "color" % (i + 1) % "png").str());
        // cv::Mat image_depth = cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1);
        cv::Mat image_color = cv::imread("/home/xhd/视频/my_vslam/slambooks14-exercise/rgbd/color/Color0000.png");
        cv::Mat image_depth = cv::imread("/home/xhd/视频/my_vslam/slambooks14-exercise/rgbd/depth/Depth0000.png", -1);
        colorImgs.push_back(image_color);
        depthImgs.push_back(image_depth); // 使用-1读取原始图像

        // 构建位姿;
        double data[7] = {0};//  x y z qx qy qz qw
        for (auto &d:data)
            fin >> d;
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    // }

    
 #if 0
    // 一些相机内参和外参
    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) <<  605.621, 0.0, 639.043, 0.0, 605.447, 368.057, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> ( 8,1 ) << 0.371033, -2.59868, -0.00029183,0.000685711, 1.59358,0.25259,-2.41826,1.51425);
    const cv::Mat K_depth = ( cv::Mat_<double> ( 3,3 ) << 605.621, 0.0, 639.043, 0.0, 605.447, 368.057, 0.0, 0.0, 1.0 );
    const cv::Mat D_depth = ( cv::Mat_<double> ( 8,1 ) << 0.371033, -2.59868, -0.00029183,0.000685711, 1.59358,0.25259,-2.41826,1.51425);


    const int ImgWidth  = image_color.cols;
    const int ImgHeight = image_color.rows;

    cv::Mat map1, map2;
    cv::Mat map1_depth, map2_depth;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 0;
    
    //                                               内参, 畸变系数, 图像尺寸,比例因子,输出图像尺寸, 输出感兴趣区域设置;
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
    cv::initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    cv::Mat NewCameraMatrix_depth = getOptimalNewCameraMatrix(K_depth, D_depth, imageSize, alpha, imageSize, 0);
    cv::initUndistortRectifyMap(K_depth, D_depth, cv::Mat(), NewCameraMatrix_depth, imageSize, CV_16SC2, map1_depth, map2_depth);


    cv::Mat UndistortImage;
    cv::Mat UndistortImage_depth;
    // cv::undistort(image, UndistortImage, K, D, K);
    cv::remap(image_color, UndistortImage, map1, map2, cv::INTER_LINEAR);
    cv::remap(image_depth, UndistortImage_depth, map1_depth, map2_depth, cv::INTER_LINEAR);
    cv::imshow("UndistortImage", UndistortImage);
    cv::waitKey(0);
    cv::imshow("UndistortImage_depth", UndistortImage_depth);
    cv::waitKey(0);
#endif
    

    // 计算点云并拼接
    // 相机内参 
    // double cx = 331.589934;
    // double cy = 326.136787;
    // double fx = 510.891164;
    // double fy = 511.345583;
    double cx = 639.043;
    double cy = 368.057;
    double fx = 605.621;
    double fy = 605.447;
    double depthScale = 1000.0;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 1; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        // cv::Mat color;
        // UndistortImage.copyTo(color); 
        // cv::Mat depth;
        // UndistortImage_depth.copyTo(depth);

        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];   // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
            }
    }

    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
