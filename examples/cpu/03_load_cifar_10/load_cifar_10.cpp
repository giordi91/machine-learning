#include <mg_ml/common/dataset.h>
#include <mg_ml/common/matrix.h>
#include <string>
#include <vector>

using core::Matrix;
int main() {
  const std::string path{"dummy"};
  Matrix X;
  Matrix Y;
  dataset::Vf Xsto;
  dataset::Vf Ysto;

  dataset::load_cifar_10(path, X, Y, Xsto, Ysto);

  return 0;
}
