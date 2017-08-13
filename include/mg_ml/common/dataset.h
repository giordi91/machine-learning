#pragma once
#include <vector>

#include <mg_ml/cpu/matrix.h>

namespace dataset {
using Vf = std::vector<float>;

void load_cifar_10(Matrix &X, Matrix &Y, Vf &Xstorage, Vf &Ystorage);
} //end namespace dataset
