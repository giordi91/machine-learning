#include <iostream>

#include <mg_ml/common/dataset.h>
#include <mg_ml/common/matrix.h>
#include <mg_ml/common/plotting.h>

int main()
{

  // loading dataset

  const std::string rootpath{"tests/datasets/coursera/cat/"};
  core::Matrix<uint8_t> X;
  core::Matrix<uint8_t> Y;
  std::vector<uint8_t> Xsto;
  std::vector<uint8_t> Ysto;

  bool res = dataset::load_coursera_cat(rootpath, X, Y, Xsto, Ysto);
  if (!res) {
    std::cout << "ERROR, could not open coursera cat at path: " << rootpath
              << std::endl;
    return 0;
  }

  std::cout << "X size: " << X.size_x << " " << X.size_y << std::endl;
  std::cout << "Y size: " << Y.size_x << " " << Y.size_y << std::endl;

  int L = 25;
  const std::string path{"/home/giordi/test.txt"};
  dataset::dump_image_from_coursera_cat_dataset(path, X,L);
  std::cout<<"Y " <<static_cast<int>(Y.data[L])<<std::endl;

  plot::GnuPlot p;
  p.name = "Cat test load";
  p.files.emplace_back(plot::plot_image(path));
  //p.show();

  //lets normalize the data
  std::vector<float> Xnormsto;
  Xnormsto.resize(X.total_size());
  core::Matrix<float> Xnorm{Xnormsto.data(), X.size_x, X.size_y};
  
  dataset::normalize_image_dataset(X, Xnorm, 255.0f);

  return 0;
}
