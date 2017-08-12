#pragma once

#include <mg_ml/cpu/matrix.h>

namespace core {
namespace cpu {
void linear_regression(const Matrix &mx, const Matrix &my, uint32_t size,
                       float learning_rate, Matrix &result, uint32_t iterations);

} // end namespace cpu
} // end namespace core
