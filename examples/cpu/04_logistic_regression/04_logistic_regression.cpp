#include <iostream>

#include <mg_ml/common/dataset.h>
#include <mg_ml/common/matrix.h>
#include <mg_ml/common/plotting.h>
#include <mg_ml/cpu/models/logistic_regression.h>
#include <random>
int main()
{

  // loading dataset

  std::cout<<"========================================================"<<std::endl;
  std::cout<<"loading train data"<<std::endl;
  const std::string rootpath{"tests/datasets/coursera/cat/"};
  core::Matrix<uint8_t> X;
  core::Matrix<uint8_t> Y;
  std::vector<uint8_t> Xsto;
  std::vector<uint8_t> Ysto;

  bool res = dataset::load_coursera_cat(rootpath, X, Y, Xsto, Ysto, true);
  if (!res) {
    std::cout << "ERROR, could not open coursera cat at path: " << rootpath
              << std::endl;
    return 0;
  }

  std::cout << "X size: " << X.size_x << " " << X.size_y << std::endl;
  std::cout << "Y size: " << Y.size_x << " " << Y.size_y << std::endl;

  int L = 0;
  const std::string path{"/home/giordi/test.txt"};
  dataset::dump_image_from_coursera_cat_dataset(path, X,L);

  plot::GnuPlot p;
  p.name = "Cat test load";
  p.files.emplace_back(plot::plot_image(path));
  //p.show();

  std::cout<<"========================================================"<<std::endl;
  std::cout<<"training ... "<<std::endl;
  //lets normalize the data
  std::vector<float> Xnormsto;
  Xnormsto.resize(X.total_size());
  core::Matrix<float> Xnorm{Xnormsto.data(), X.size_x, X.size_y};
  std::vector<float> Ynormsto;
  Ynormsto.resize(Y.total_size());
  core::Matrix<float> Ynorm{Ynormsto.data(), Y.size_x, Y.size_y};
  
  dataset::normalize_image_dataset(X, Xnorm, 255.0f);
  dataset::normalize_image_dataset(Y, Ynorm, 1.0f);
  

  float learning_rate = 0.005;
  uint32_t iterations = 2000;
  
  //initializing the weights
  std::vector<float> Wstorage(X.size_y);
  core::Matrix<float> Wm{Wstorage.data(), 1, X.size_y};
  core::cpu::initialize_to_zeros(Wm);
  models::cpu::simple_logistic_model<float>(Xnorm, Wm, Ynorm, iterations, learning_rate);

  std::vector<float> Ypredict;
  Ypredict.resize(X.total_size());
  core::Matrix<float> YpredictM{Ypredict.data(), Y.size_x, Y.size_y};

  float train_acc=0.0f; 
  train_acc = models::cpu::logistic_model_predict(Xnorm, Wm, Ynorm, YpredictM);
  std::cout << "accuracy of the training set is " << train_acc << "%"
            << std::endl;

  //testing against cross validation test
  core::Matrix<uint8_t> Xtest;
  core::Matrix<uint8_t> Ytest;
  std::vector<uint8_t> Xteststo;
  std::vector<uint8_t> Yteststo;

  std::cout<<"========================================================"<<std::endl;
  std::cout<<"loading test data "<<std::endl;
  res = dataset::load_coursera_cat(rootpath, Xtest, Ytest, Xteststo, Yteststo, true, true);
  if (!res) {
    std::cout << "ERROR, could not open coursera test cat at path: " << rootpath
              << std::endl;
    return 0;
  }

  std::cout << "Xtest size: " << Xtest.size_x << " " << Xtest.size_y << std::endl;
  std::cout << "Ytest size: " << Ytest.size_x << " " << Ytest.size_y << std::endl;

  //normalize 
  //lets normalize the data
  std::vector<float> Xtestnormsto;
  Xtestnormsto.resize(Xtest.total_size());
  core::Matrix<float> Xtestnorm{Xtestnormsto.data(), Xtest.size_x, Xtest.size_y};
  std::vector<float> Ytestnormsto;
  Ytestnormsto.resize(Ytest.total_size());
  core::Matrix<float> Ytestnorm{Ytestnormsto.data(), Ytest.size_x, Ytest.size_y};
  
  dataset::normalize_image_dataset(Xtest, Xtestnorm, 255.0f);
  dataset::normalize_image_dataset(Ytest, Ytestnorm, 1.0f);

  std::vector<float> Ypredicttest;
  Ypredicttest.resize(Xtest.total_size());
  core::Matrix<float> YpredicttestM{Ypredicttest.data(), Ytest.size_x, Ytest.size_y};
  float test_acc=0.0f; 
  test_acc = models::cpu::logistic_model_predict(Xtestnorm, Wm, Ytestnorm, YpredicttestM);

  std::cout << "accuracy of the test set is " << test_acc <<"%"<< std::endl;

  return 0;
}
