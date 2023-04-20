#include <chrono>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include <sophus/se3.hpp>

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
// 自实现节点同时优化相机位姿,与三维点位置;
void bundleAdjustmentG2O(const VecVector3d &points_3d,
                         const VecVector2d &points_2d_1,
                         const VecVector2d &points_2d_2, 
                         const cv::Mat &K,
                         Sophus::SE3d &pose);
// 用g2o带节点同时优化相机位姿和n三维点位置;
void bundleAdjustmentG2O_2(const std::vector<cv::Point3f> &points_3d,
                         const std::vector<cv::Point2f> &points_2d_1,
                         const std::vector<cv::Point2f> &points_2d_2, 
                         const cv::Mat &K,
                         Sophus::SE3d &pose);

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2"
              << std::endl;
    return 1;
  }

  cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  std::vector<cv::DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  std::cout << "一共找到了" << matches.size() << "组匹配点!" << std::endl;

  cv::Mat d1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  std::vector<cv::Point3f> pts_3d;
  std::vector<cv::Point2f> pts_2d_2;
  std::vector<cv::Point2f> pts_2d_1;
  for (cv::DMatch m : matches) {
    ushort d = d1.ptr<ushort>(
        int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)
      continue;
    float dd = d / 5000.0;
    cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d_2.push_back(keypoints_2[m.trainIdx].pt);
    pts_2d_1.push_back(keypoints_1[m.queryIdx].pt);
  }
  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen_2;
  VecVector2d pts_2d_eigen_1;
  for (size_t i = 0; i < pts_3d.size(); i++) {
    pts_3d_eigen.push_back(
        Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen_2.push_back(Eigen::Vector2d(pts_2d_2[i].x, pts_2d_2[i].y));
    pts_2d_eigen_1.push_back(Eigen::Vector2d(pts_2d_1[i].x, pts_2d_1[i].y));
  }
  std::cout << "3d-2d pairs: " << pts_3d.size() << std::endl;
  std::cout << "calling bundle adjustment by g2o" << std::endl;

  Sophus::SE3d pose_g2o;

  // bundleAdjustmentG2O_2(pts_3d, pts_2d_1, pts_2d_2, K, pose_g2o);
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen_1, pts_2d_eigen_2, K, pose_g2o);

  return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
  //-- 初始化
  cv::Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB"
  // );
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  std::vector<cv::DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离,
  //即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                     (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
//顶点构造
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4],update[5];
    // for (int i = 0; i < 6; ++i) {
    //   std::cout << update[i] << std::endl;
    //   if (std::isnan(update[i])) {
    //     update_eigen[i] = 0;
    //     std::cout << "reslut is nan!" << std::endl;
    //   }
    // }
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(std::istream &in) override {}
  virtual bool write(std::ostream &out) const override {}
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override {
    _estimate =
        Eigen::Vector3d(0, 0, 0); //这个地方是当前的三维点初始所在位置?todo:
  }
  virtual void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
  }
  virtual bool read(std::istream &in) override {}
  virtual bool write(std::ostream &out) const override {}
  // g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
};
//边
class EdgeProjection: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose,
                                 VertexPoint> { // todo: 误差是2还是3?
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Vector3d &point, const Eigen::Matrix3d &K)
      : _point_3d(point), _K(K) {} // 投影是像素误差的时候需要这么写;

  virtual void computeError() override {
    //_vertices[0]与_vertices[1]这个中0和1是不是决定了_jacobianOplusXi和_jacobianOplusXj的顺序;
    const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
    // const VertexPoint *point = static_cast<const VertexPoint *>(_vertices[1]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d pose_pixel = _K * (T * _point_3d);
    pose_pixel /= pose_pixel[2];
    _error = _measurement - pose_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _point_3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    // TODO:此处关于雅克比矩阵的前三列和后三列的顺序,高翔的原版程序是不应该是错的?
    // 应该对调三列与后三列的位置.
    //因为G2O中李代数的定义方式是旋转在前平移在后;而高翔在书中的关于李代数se(3)推导是平移在前的;岂不是错了?
    Eigen::Matrix<double, 2, 3> tmp;
    tmp(0, 0) = -fx / Z;
    tmp(0, 1) = 0;
    tmp(0, 2) = fx * X / Z2;
    tmp(1, 0) = 0;
    tmp(1, 1) = -fy / Z;
    tmp(1, 2) = fy * Y / (Z * Z);
    // 误差关于位姿的偏导
    // 误差关于世界坐标系下坐标点的偏导

    _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2,-fx - fx * X * X / Z2, fx * Y / Z, 
                        0, -fy / Z, fy * Y / (Z * Z),fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    _jacobianOplusXj = tmp * T.so3().matrix();
  }
  virtual bool read(std::istream &in) override {}
  virtual bool write(std::ostream &out) const override {}

private:
  Eigen::Vector3d _point_3d;
  Eigen::Matrix3d _K;
};
#if 1
void bundleAdjustmentG2O(const VecVector3d &points_3d,
                         const VecVector2d &points_2d_1,
                         const VecVector2d &points_2d_2, 
                         const cv::Mat &K,
                         Sophus::SE3d &pose) {

  // typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>BlockSolverType; //指定pose维度为6,landmark维度为3
  // typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>LinearSolverType; // 线性求解器类型
  // auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  // g2o::SparseOptimizer optimizer;
  // optimizer.setAlgorithm(solver);
  // optimizer.setVerbose(true);

  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; //指定pose维度为6,landmark维度为3
  typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);


  // 第一个相机位姿顶点;
  VertexPose *poseOne = new VertexPose();
  poseOne->setId(0);
  poseOne->setFixed(1);
  poseOne->setEstimate(Sophus::SE3d());
  optimizer.addVertex(poseOne);

  // 第二个相机i位姿顶点;
  VertexPose *poseTwo = new VertexPose();
  poseTwo->setId(1);
  poseTwo->setEstimate(Sophus::SE3d());
  optimizer.addVertex(poseTwo);

  // 三维空间点顶点;
  int index = 2;
  for (const auto p : points_3d) {
    VertexPoint *point = new VertexPoint();
    point->setId(index++);
    point->setEstimate(p);
    point->setMarginalized(true);
    optimizer.addVertex(point);
  }

  //对相机内参进行优化
  g2o::CameraParameters *camera = new g2o::CameraParameters(
      K.at<double>(0, 0),
      Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
  camera->setId(0);
  optimizer.addParameter(camera);

  // 第一个相机的观测
  int edgeCount = 0;
  Eigen::Matrix3d _K;
  cv::cv2eigen(K, _K);
  index = 2;
  for (int i = 0; i < points_3d.size(); ++i) {
    EdgeProjection *edgeOne = new EdgeProjection(points_3d[i], _K);
    edgeOne->setId(edgeCount++);
    //链接两个顶点
    edgeOne->setVertex(0, dynamic_cast<VertexPose *>(optimizer.vertex(0)));
    edgeOne->setVertex(1, dynamic_cast<VertexPoint *>(optimizer.vertex(index))); //注意
    //测量值为第一帧的像素坐标
    edgeOne->setMeasurement(points_2d_1[i]);
    edgeOne->setParameterId(0,0);
    edgeOne->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edgeOne);
    index++;
  }
  //第二个相机的观测
  index = 2;
  for (int i = 0; i < points_3d.size(); ++i) {
    EdgeProjection *edgeTwo = new EdgeProjection(points_3d[i], _K);
    edgeTwo->setId(edgeCount++);
    edgeTwo->setVertex(0, dynamic_cast<VertexPose *>(optimizer.vertex(1)));
    edgeTwo->setVertex(1,dynamic_cast<VertexPoint *>(optimizer.vertex(index)));
    edgeTwo->setMeasurement(points_2d_2[i]);
    edgeTwo->setParameterId(0, 0);
    edgeTwo->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edgeTwo);
    index++;
  }

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(20);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "optimization costs time: " << time_used.count()*1000 << " ms." << std::endl;
  std::cout << "pose estimated by g2o =\n" << poseTwo->estimate().matrix() << std::endl;
  pose = poseTwo->estimate();  // 返回值;
};
#endif
#if 1
void bundleAdjustmentG2O_2(const std::vector<cv::Point3f> &points_3d,
                         const std::vector<cv::Point2f> &points_2d_1,
                         const std::vector<cv::Point2f> &points_2d_2, 
                         const cv::Mat &K,
                         Sophus::SE3d &pose) {

  // typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  //优化位姿6维  优化路标点3维
  // std::unique_ptr<Block::LinearSolverType> linearSolver=g2o::make_unique < g2o::LinearSolverCSparse<Block::PoseMatrixType> >();//线性求解设为CSparse
  // std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver) ) );
  // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );

  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; //指定pose维度为6,landmark维度为3
  typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  // 第一个相机位姿顶点;
  g2o::VertexSE3Expmap* poseOne = new g2o::VertexSE3Expmap();
  poseOne->setId(0);  
  poseOne->setFixed(1); //固定
  poseOne->setEstimate(g2o::SE3Quat()); 
  optimizer.addVertex(poseOne);

  // 第二个相机i位姿顶点;
  g2o::VertexSE3Expmap* poseTwo = new g2o::VertexSE3Expmap();
  poseTwo->setId(1);// 1
  poseTwo->setEstimate(g2o::SE3Quat());//构造李代数;
  optimizer.addVertex(poseTwo);

  // 三维空间点顶点;
  int index = 2; //2
  for (int i = 0; i < points_3d.size(); i++) {
    g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
    point->setId(index++);
    point->setMarginalized(true);
    point->setEstimate(Eigen::Vector3d(points_3d[i].x, points_3d[i].y, points_3d[i].z));
    optimizer.addVertex(point);
  }

  //对相机内参进行优化
  g2o::CameraParameters *camera = new g2o::CameraParameters(
      K.at<double>(0, 0),Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
  camera->setId(0);
  optimizer.addParameter(camera);

  // 第一个相机的观测
  int edgeCount = 0;
  Eigen::Matrix3d _K;
  cv::cv2eigen(K, _K);
  
  //添加边;
  index = 2;
  for (int i = 0; i < points_2d_1.size(); ++i) {
    g2o::EdgeProjectXYZ2UV* edgeOne = new g2o::EdgeProjectXYZ2UV();
    edgeOne->setId(edgeCount++);
    //链接两个顶点
    edgeOne->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index++)));//0对应三维点;
    edgeOne->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)));//第一个相机位姿;
    //测量值为第一帧的像素坐标
    edgeOne->setMeasurement(Eigen::Vector2d(points_2d_1[i].x,points_2d_1[i].y));//此处是不不能是const修饰的点?不是
    edgeOne->setParameterId(0,0);
    edgeOne->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edgeOne);
  }
  //第二个相机的观测
  index = 2;//注意;  //2
  for (int i = 0; i < points_2d_2.size(); ++i) {
    g2o::EdgeProjectXYZ2UV* edgeOne = new g2o::EdgeProjectXYZ2UV();
    edgeOne->setId(edgeCount++);
    //链接两个顶点
    edgeOne->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index++)));
    edgeOne->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)));
    //测量值为第一帧的像素坐标
    edgeOne->setMeasurement(Eigen::Vector2d(points_2d_2[i].x,points_2d_2[i].y));
    edgeOne->setParameterId(0,0);
    edgeOne->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edgeOne);
  }
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "optimization costs time :" << time_used.count() << "s."<< std::endl;
    g2o::SE3Quat pose_1 = poseOne->estimate();
  std::cout << "pose estimated of camera1 by g2o =\n"<< pose_1 << std::endl;
  std::cout << "***************************************************************"<< std::endl;
    g2o::SE3Quat pose_2 = poseTwo->estimate();
  std::cout << "pose estimated of camera2 by g2o =\n"<< pose_2 << std::endl;
  std::cout << "***************************************************************"<< std::endl;
  g2o::SE3Quat pose_diff = poseOne->estimate().inverse() *poseTwo->estimate(); //此时待求的两个相机之间的相对位姿
  std::cout << "pose estimated by g2o =\n" << pose_diff << std::endl;
  std::cout << "***************************************************************" << std::endl;
};
#endif