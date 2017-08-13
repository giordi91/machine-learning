#pragma once
#include <vector>
#include <string>

#include <mg_ml/common/matrix.h>

namespace dataset {
using core::MatrixI;

bool load_cifar_10(const std::string &rootpath, MatrixI<char> &X, MatrixI<char> &Y,
                   std::vector<char> &Xstorage, std::vector<char> &Ystorage);

} // end namespace dataset
