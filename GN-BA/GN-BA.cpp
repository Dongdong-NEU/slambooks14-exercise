//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.hpp"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "./../GN-BA/p3d.txt"; // 相机空间坐标;
string p2d_file = "./../GN-BA/p2d.txt"; // 像素坐标还是归一化平面上的坐标?

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    ifstream fin_p3d(p3d_file);
    ifstream fin_p2d(p2d_file);
    if(!fin_p2d.is_open() || !fin_p3d.is_open()){
        std::cout << "txt file is not open!" << std::endl;
        return 1;
    }
    double data[3] = {0};
    while(!fin_p2d.eof()){
        for(int i = 0; i < 2; i++){
            fin_p2d >> data[i];
        }
        Eigen::Vector2d item(data[0], data[1]);
        p2d.push_back(item);
    }
    while(!fin_p3d.eof()){
        for(int i = 0; i < 3; i++){
            fin_p3d >> data[i];
        }
        Eigen::Vector3d item(data[0], data[1], data[2]);
        p3d.push_back(item);
    }
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3d T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
        // compute cost for p3d[I] and p2d[I]
        // START YOUR CODE HERE 
            Eigen::Vector4d P_ = T_esti.matrix() * Eigen::Vector4d(p3d[i](0,0), p3d[i](1,0), p3d[i](2,0), 1);
            Eigen::Vector3d u = K * Eigen::Vector3d(P_(0,0),P_(1,0),P_(2,0));
            Eigen::Vector2d e = p2d[i] - Eigen::Vector2d(u(0,0)/u(2,0), u(1,0)/u(2,0)); //e = p2d - K*T*p3d
            cost += e.squaredNorm() / 2; //cost即最小二乘的表达式 cost += e^2/2;
	    // END YOUR CODE HERE

	    // compute jacobian
            Matrix<double, 2, 6> J;
        // START YOUR CODE HERE 
            double x = P_(0,0), y = P_(1,0), z = P_(2,0);
            double x2 = x * x, y2 = y * y, z2 = z *z;
            J(0, 0) = fx/z;
            J(0, 1) = 0;
            J(0, 2) = -fx*x/z2;
            J(0, 3) = -fx*x*y/z2;
            J(0, 4) = fx + fx*x2/z2;
            J(0, 5) = -fx*y/z;

            J(1, 0) = 0;
            J(1, 1) = fy/z;
            J(1, 2) = - fy * y / z2;
            J(1, 3) = -fy - fy*y2/z2;
            J(1, 4) = fy*x*y/z2;
            J(1, 5) = fy * x / z;
            J=-J;   ////注意此处;
	    // END YOUR CODE HERE
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

	// solve dx 
        Vector6d dx;

        // START YOUR CODE HERE 
        dx = H.ldlt().solve(b);
        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 
        T_esti = Sophus::SE3d::exp(dx) * T_esti;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
