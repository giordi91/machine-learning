#pragma once
#include <cmath>

#include <mg_ml/common/matrix.h>


namespace models {
namespace cpu {

using core::Matrix;

template<typename T> 
void sigmoid(const Matrix<T>& in, Matrix<T>& out)
{

  assert(in.size_x == out.size_x);
  assert(in.size_y == out.size_y);

  uint32_t total_size = in.total_size();
  const T *const iptr = in.data;
  T *const optr = out.data;
  for (uint32_t i = 0; i < total_size; ++i) {
    optr[i] = 1.0f / (1 + exp(-iptr[i]));
  }
}

template<typename T> 
void sigmoid_inplace(const Matrix<T>& m)
{

  uint32_t total_size = m.total_size();
  T *const mptr = m.data;
  for (uint32_t i = 0; i < total_size; ++i) {
    mptr[i] = 1.0f / (1 + exp(-mptr[i]));
  }
}
} // end namespace cpu
} // end namespace models 
