#pragma once
#include <vector>
#include <string>

#include <mg_ml/common/matrix.h>

namespace dataset {
using Vf = std::vector<float>;
using core::Matrix;

void load_cifar_10(const std::string &path, Matrix &X, Matrix &Y, Vf &Xstorage,
                   Vf &Ystorage);
} // end namespace dataset
