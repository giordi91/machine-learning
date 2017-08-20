#pragma once
#include <vector>
#include <string>

#include <mg_ml/common/matrix.h>

namespace dataset {
using core::Matrix;
//This functions are used for the cifar 10 dataset which can be found here
// https://www.cs.toronto.edu/~kriz/cifar.html
//The dataset has the following 10 classes
enum class Cifar10_classes
{
    AIRPLANE=0,
    AUTOMOBILE,
    BIRD,
    CAT,
    DEER,
    DOG,
    FROG,
    HORSE,
    SHIP,
    TRUCK
};

bool load_cifar_10(const std::string &rootpath, Matrix<uint8_t> &X, Matrix<uint8_t> &Y,
                   std::vector<uint8_t> &Xstorage, std::vector<uint8_t> &Ystorage);
bool dump_image_from_cifar_10_dataset(const std::string &outpath,
                                      Matrix<uint8_t> &data, uint32_t index);

//coursera
bool load_coursera_cat(const std::string &outpath, Matrix<uint8_t> &X,
                      Matrix<uint8_t> &Y, std::vector<uint8_t> &Xstorage,
                      std::vector<uint8_t> &Ystorage);

bool dump_image_from_coursera_cat_dataset(const std::string &outpath,
                                      Matrix<uint8_t> &data, uint32_t index);

void normalize_image_dataset(Matrix<uint8_t> &data, Matrix<float> &dataout,
                             float norm_value);

} // end namespace dataset
