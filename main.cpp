#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

class A
{
public:
    A(const int &i) : index(i) {}
    int index = 0;
};

int main()
{
// 1. eigen中矩阵和向量的初始化；
// 2. 基本运算，易错之处；
// 3. 求解方程Ax = b A为100*100的随机矩阵，查看QR分解与Cholesky分解的速度，解释为什么？
#if 0    
    // Eigen基本运算部分；
    Eigen::Matrix<float, 2, 3> m_23f;
    Eigen::Matrix<float, 3, 1> m_31f;
    Eigen::Vector3d v_3d;
    Eigen::Vector3f v_3f;
    m_23f << 1,2,3,4,5,6;
    m_31f << 1,2,3;
    v_3d << 1,2,3;
    v_3f << 1,2,3;

    Eigen::Matrix3d m_33d = Eigen::Matrix3d::Random();
    cout << m_33d << endl;

    Eigen::Matrix<double,3,1> mres_31d = v_3d + v_3f.cast<double>();
    std::cout << "mres_21f : \n" << mres_31d << std::endl;

    //计算矩阵的特征值与特征向量；
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(m_33d.transpose()*m_33d);
    std::cout << "Eigen value = \n" << eigen_solver.eigenvalues() << std::endl;
    std::cout << "Eigen vector = \n" << eigen_solver.eigenvectors() << std::endl;

    //解方程
    Eigen::Matrix<double, 100, 100> m_100100d = Eigen::MatrixXd::Random(100,100);
    m_100100d = m_100100d * m_100100d.transpose();
    Eigen::Matrix<double, 100,1> m_1001d = Eigen::MatrixXd::Random(100,1);
    clock_t time_sst = clock();
    Eigen::Matrix<double,100,1> mx_1001 = m_100100d.inverse() * m_1001d;
    std::cout << "time of normal inverse is "  << 1000 * (clock() - time_sst)/(double)CLOCKS_PER_SEC <<"ms" << std::endl;
    //std::cout << "mx_1001 = " << mx_1001 << std::endl;

    time_sst = clock();
    mx_1001 = m_100100d.colPivHouseholderQr().solve(m_1001d);
    std::cout << "time of QR decomposition is "  << 1000 * (clock() - time_sst)/(double)CLOCKS_PER_SEC <<"ms" << std::endl;
    //std::cout << "mx_1001 = " << mx_1001 << std::endl;

    time_sst = clock();
    mx_1001 = m_100100d.ldlt().solve(m_1001d);
    std::cout << "time of ldlt decomposition is "  << 1000 * (clock() - time_sst)/(double)CLOCKS_PER_SEC <<"ms" << std::endl;
    //std::cout << "mx_1001 = " << mx_1001 << std::endl;
#endif
#if 0
    // 四元数转换旋转矩阵；
    Eigen::Quaterniond q1(0.55,0.3,0.2,0.2),q2(-0.1,0.3,-0.7,0.2);
    Eigen::Vector3d t1(0.7,1.1,0.2),t2(-0.1,0.4,0.8);
    Eigen::Vector3d p1(0.5,-0.1,0.2);
    q1.normalize();
    q2.normalize();
    Eigen::Isometry3d T1w(q1), T2w(q2);
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);
    Eigen::Vector3d p2 = T2w * T1w.inverse() * p1;
    std::cout << "p2: " << endl;
    std::cout << p2.transpose() << endl;
#endif

#if 0
    // C++11 的一些新特性；
    A a1(3), a2(5), a3(9), a4(0);
    std::cout << a4.index << std::endl;
    vector<A> avec{a1, a2, a3};
    std::sort(avec.begin(), avec.end(), [](const A &a1, const A &a2)
              { return a1.index < a2.index; });
    for (auto &a : avec)
        cout << a.index << " ";
    cout << endl;
#endif
        return 0;
}
