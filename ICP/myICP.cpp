//用ICP实现轨迹对齐
//把两条轨迹的平移部分看作点集,然后求点集之间的 ICP,得到两组点之间的变换T_eg

#include <sophus/se3.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;
void ReadData(string FileName ,
              vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses_e,
              vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses_g,
              vector<Point3f> &t1,
              vector<Point3f> &t2
);
void icp_svd (
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Matrix3d & R,Vector3d& t);
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e,
                    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g);

/*******************************main函数*************************************/
int main(int argc,char **argv){
    string TrajectoryFile = "./../ICP/compare.txt";
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g_; //poses_g_=T*poses_g
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    vector<Point3f> t_e,t_g;

    ReadData( TrajectoryFile,poses_e, poses_g, t_e, t_g);
    icp_svd(t_e,t_g,R,t);
    Sophus::SE3d T_eg(R,t);
    for(auto SE3_g:poses_g){
        SE3_g =T_eg*SE3_g; // T_e[i]=T_eg*T_g[i]
        poses_g_.push_back(SE3_g);
    }
   DrawTrajectory(poses_e,poses_g_);
}
/*************读取文件中的位姿******************/
void ReadData(string FileName ,
        vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses_e,
        vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> &poses_g,
        vector<Point3f> &t_e,
        vector<Point3f> &t_g
        ){
    string line;
    double time1,tx_1,ty_1,tz_1,qx_1,qy_1,qz_1,qw_1;
    double time2,tx_2,ty_2,tz_2,qx_2,qy_2,qz_2,qw_2;
    ifstream fin(FileName);
    if(!fin.is_open()){
        cout<<"compare.txt file can not open!"<<endl;
        return ;
    }
    while(getline(fin,line)){
        istringstream record(line);
        record>>time1 >> tx_1 >> ty_1 >> tz_1 >> qx_1 >> qy_1 >> qz_1 >> qw_1
              >>time2 >> tx_2 >> ty_2 >> tz_2 >> qx_2 >> qy_2 >> qz_2 >> qw_2;
        t_e.push_back(Point3d(tx_1,ty_1,tz_1)); //将t取出，为了进行用icp进行计算
        t_g.push_back(Point3d(tx_2,ty_2,tz_2));

        Eigen::Vector3d point_t1(tx_1, ty_1, tz_1);
        Eigen::Vector3d point_t2(tx_2, ty_2, tz_2);

        Eigen::Quaterniond q1 = Eigen::Quaterniond(qw_1, qx_1, qy_1, qz_1).normalized(); //四元数的顺序要注意
        Eigen::Quaterniond q2 = Eigen::Quaterniond(qw_2, qx_2, qy_2, qz_2).normalized();

        Sophus::SE3d SE3_qt1(q1, point_t1);
        Sophus::SE3d SE3_qt2(q2, point_t2);

        poses_e.push_back(SE3_qt1);
        poses_g.push_back(SE3_qt2);
    }

}

void icp_svd (
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Matrix3d& R, Vector3d& t) {
    Point3f p1, p2; // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) / N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f> q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    R = U* ( V.transpose() ); //p1=R_12*p_2,注意R的意义，p2到p1的旋转关系
    t = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R * Eigen::Vector3d ( p2.x, p2.y, p2.z );
}

/*****************************绘制轨迹*******************************************/
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e,
                    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g) {
    if (poses_g.empty() || poses_e.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768); //创建一个窗口
    glEnable(GL_DEPTH_TEST); //启动深度测试
    glEnable(GL_BLEND); //启动混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //混合函数glBlendFunc( GLenum sfactor , GLenum dfactor );sfactor 源混合因子dfactor 目标混合因子

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0) //对应的是gluLookAt,摄像机位置,参考点位置,up vector(上向量)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses_e.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_e.size(), 0.0f, (float) i / poses_e.size());
            glBegin(GL_LINES);
            auto p1 = poses_e[i], p2 = poses_e[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        for (size_t j = 0; j < poses_g.size() - 1; j++) {
            glColor3f(1 - (float) j / poses_g.size(), 0.0f, (float) j / poses_g.size());
            glBegin(GL_LINES);
            auto p1 = poses_g[j], p2 = poses_g[j + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        pangolin::FinishFrame();
    }

}

