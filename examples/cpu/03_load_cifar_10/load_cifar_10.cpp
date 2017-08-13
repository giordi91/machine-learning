#include <iostream>
#include <string>
#include <vector>

#include <mg_ml/common/dataset.h>
#include <mg_ml/common/matrix.h>

using core::MatrixI;
int main() {
  const std::string rootpath{"tests/datasets/"};
  core::MatrixI<char> X;
  core::MatrixI<char> Y;
  std::vector<char> Xsto;
  std::vector<char> Ysto;

  bool res = dataset::load_cifar_10(rootpath, X, Y, Xsto, Ysto);
  if (!res) {
    std::cout << "ERROR, could not open cifar_10 at path: " << rootpath
              << std::endl;
    return 0;
  }

  std::cout<<"X size: "<<X.size_x<< " "<<X.size_y<<std::endl;
  std::cout<<"Y size: "<<Y.size_x<< " "<<Y.size_y<<std::endl;

  return 0;
}
