#include <iostream>
#include <string>
#include <vector>

#include <mg_ml/common/dataset.h>
#include <mg_ml/common/matrix.h>

int main() {
  const std::string rootpath{"tests/datasets/"};
  core::Matrix<uint8_t> X;
  core::Matrix<uint8_t> Y;
  std::vector<uint8_t> Xsto;
  std::vector<uint8_t> Ysto;

  bool res = dataset::load_cifar_10(rootpath, X, Y, Xsto, Ysto);
  if (!res) {
    std::cout << "ERROR, could not open cifar_10 at path: " << rootpath
              << std::endl;
    return 0;
  }

  std::cout<<"X size: "<<X.size_x<< " "<<X.size_y<<std::endl;
  std::cout<<"Y size: "<<Y.size_x<< " "<<Y.size_y<<std::endl;

  int L = 3;
  dataset::dump_image_from_cifar_10_dataset("/home/giordi/test.txt", X,L);
  std::cout<<"Y " <<static_cast<int>(Y.data[L])<<std::endl;

  
  return 0;
}
