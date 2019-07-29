#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "hello_world.hpp"

int main(void)
{
  Eigen::MatrixXd mat(2,3);
  Eigen::VectorXd vec(3);

  mat(0, 0) = 1.0;
  mat(0, 0) = 2.0;
  mat(0, 0) = 3.0;
  mat(0, 0) = 4.0;
  mat(0, 0) = 5.0;
  mat(0, 0) = 6.0;

  vec(0) = vec(1) = vec(2) = 1.0;

  std::cout << "v.v = " << vec.dot(vec) << std::endl;

  std::cout << mat << std::endl;

  Eigen::MatrixXd mat2 = mat * mat.transpose();

  std::cout << mat << std::endl;
  std::cout << mat2 << std::endl;

  Eigen::MatrixXd vmult = mat * vec;
  std::cout << "vmult" << std::endl << vmult << std::endl;

  std::cout << "rows = " << mat.rows() << std::endl;
  std::cout << "cols = " << mat.cols() << std::endl;


  return 0;
}
