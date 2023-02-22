#include <sophus/se3.hpp>
#include <string>
#include <iostream>
#include <fstream>
// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>
#include <Eigen/Core>

using namespace std;

string estimated_file = "/home/xhd/视频/my_vslam/slambooks14-exercise/trajectory/estimated.txt";
string groundtruth_file = "/home/xhd/视频/my_vslam/slambooks14-exercise/trajectory/groundtruth.txt";

typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
TrajectoryType ReadTrajectory(const string &path);
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

int main(int argc, char **argv) {

    /// implement pose reading code
    TrajectoryType estimated = ReadTrajectory(estimated_file);
    TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);

    assert(!groundtruth.empty() && !estimated.empty());
    assert( groundtruth.size() == estimated.size());  // 不符合内部的条件的时候,退出;
    
    double rmse = 0;
    for(size_t i = 0; i < groundtruth.size(); i++){
        Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
        double err = (p2.inverse() * p1).log().norm();
        rmse += err * err;
    }

    rmse = rmse / estimated.size();
    rmse = sqrt(rmse);

    std::cout << "RMSE =" << rmse << std::endl;

    // draw trajectory in pangolin
    DrawTrajectory(groundtruth,estimated);
    return 0;
}

TrajectoryType ReadTrajectory(const string &path){
    ifstream fin(path);
    TrajectoryType trajectory;
    if( !fin ){
        std::cerr << "trajectory " << path << "not found!" << std::endl;
        return trajectory;
    }

    while( !fin.eof() ){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        trajectory.push_back(p1);
    }
    return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
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

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}