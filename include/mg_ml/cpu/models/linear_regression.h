#pragma once

#include <mg_ml/cpu/matrix_functions.h>

namespace models {
namespace cpu {

using core::Matrix;
using core::cpu::matrix_mult;
using core::cpu::matrix_mult_transpose;
using core::cpu::matrix_sub;
using core::cpu::vector_sub;
using core::cpu::matrix_mult_scalar_inplace;

template <typename T>
inline void cost_function_grad(const Matrix<T> &mx, const Matrix<T> &my,
                               const Matrix<T> &coeff, Matrix<T> outCoeff,
                               Matrix<T> &intermediate) {

  // here we compute the regular cost by computing the weights time the samples
  matrix_mult_transpose(coeff, mx, intermediate);
  // we subtract the error so we compute the error
  matrix_sub(intermediate, my, intermediate);

  // here we allocate temporary data to get the transpose of the intremediate
  // matrix
  // TODO use custom allocator to avoid expensive allocation on a per iteration
  // basis
  std::vector<float> transpose_intermediate;
  transpose_intermediate.resize(intermediate.size_y * intermediate.size_x);

  // in order to get the right matrix size we need to transpose the intermediate
  // size and
  // perform a matrix multiplication, this is not the most efficient way to do
  // it, if
  // we can transpose my, we can get away with one transpose only of mx
  Matrix<T> transp{transpose_intermediate.data(), intermediate.size_y,
                   intermediate.size_x};
  matrix_transpose(intermediate, transp);
  matrix_mult(transp, mx, outCoeff);
}

template <typename T>
inline void
cost_function_grad_y_transposed(const Matrix<T> &mx, const Matrix<T> &my,
                                const Matrix<T> &coeff, Matrix<T> outCoeff,
                                Matrix<T> &intermediate) {

  // here we compute the regular cost by computing the weights time the samples
  matrix_mult_transpose(coeff, mx, intermediate);
  // we subtract expected result so we compute the error
  // here we are using vector sub, because we expect the cost to be a vector,
  // so doesnt matter if is a colum or row vector we can subtract it correctly
  vector_sub(intermediate, my, intermediate);
  // multiplying the resulting cost by the colum feature of x
  matrix_mult(intermediate, mx, outCoeff);
}

template <typename T>
inline float cost_function(const Matrix<T> &mx, const Matrix<T> &my,
                           Matrix<T> &intermediate, uint32_t size,
                           const Matrix<T> &coeff) {
  // here we multiply the weights by the samples
  matrix_mult_transpose(mx, coeff, intermediate);
  // we subtract the error
  matrix_sub(intermediate, my, intermediate);
  // finally we accumulate the error vector
  matrix_ew_mult(intermediate, intermediate, intermediate);
  // final error is devided by twice the size of the vector
  return (1.0f / (float(size) * 2.0f)) * matrix_accumulate(intermediate);
}
template <typename T>
void linear_regression(const Matrix<T> &mx, const Matrix<T> &my, uint32_t size,
                       float learning_rate, Matrix<T> &result,
                       uint32_t iterations) {

  float learning_c = learning_rate * (1.0f / float(size));

  // TODO use custom allocator
  std::vector<float> temp_m(mx.size_x * result.size_x);
  Matrix<T> temp{temp_m.data(), result.size_x, mx.size_x};
  // TODO use custom allocator, although here is a fixed initial cost not
  // too much of an issue
  std::vector<float> outRes;
  outRes.resize(result.size_x * result.size_y);
  Matrix<T> tempRes{outRes.data(), result.size_x, result.size_y};

  for (uint32_t i = 0; i < iterations; ++i) {
    cost_function_grad_y_transposed(mx, my, result, tempRes, temp);

    // mult_by_learning_rate
    matrix_mult_scalar_inplace(tempRes, learning_c);
    // update final
    matrix_sub(result, tempRes, result);
  }
}

} // end namespace cpu
} // end namespace models 
