#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <mg_ml/common/matrix.h>
namespace core {

namespace cpu {


template <typename T> void matrix_transpose(const Matrix<T> &m, Matrix<T> &out)
{

  assert(m.size_x == out.size_y);
  assert(m.size_y == out.size_x);

  int N = m.size_x;
  int M = m.size_y;
  for (int n = 0; n < N * M; n++) {
    int i = n / N;
    int j = n % N;
    out.data[n] = m.data[M * j + i];
  }
}
template <typename T>
void matrix_mult_transpose(const Matrix<T> &m1, const Matrix<T> &m2,
                           Matrix<T> &out)
{

  assert(m1.size_y == m2.size_y);
  assert(m1.size_x == out.size_x);
  assert(m2.size_x == out.size_y);

  const float *const d1 = m1.data;
  const float *const d2 = m2.data;
  float *const o = out.data;

  for (uint32_t r1 = 0; r1 < out.size_x; ++r1) {
    for (uint32_t r2 = 0; r2 < out.size_y; ++r2) {
      float accum = 0.0f;
      for (uint32_t c1 = 0; c1 < m2.size_y; ++c1) {
        accum += d1[m1.size_y * r1 + c1] * d2[m2.size_y * r2 + c1];
      }
      o[out.size_y * r1 + r2] = accum;
    }
  }
}

template <typename T>
void matrix_mult(const Matrix<T> &m1, const Matrix<T> &m2, Matrix<T> &out)
{
  assert(m1.size_y == m2.size_x);
  assert(m1.size_x == out.size_x);
  assert(m2.size_y == out.size_y);
  std::vector<float> temp;
  temp.resize(m2.size_x * m2.size_y);
  Matrix<T> tempM{temp.data(), m2.size_y, m2.size_x};

  matrix_transpose<T>(m2, tempM);
  matrix_mult_transpose<T>(m1, tempM, out);
}


template <typename T>
void matrix_sub(const Matrix<T> &m1, const Matrix<T> &m2, Matrix<T> &out)
{

  assert(m1.size_x == m2.size_x);
  assert(m1.size_y == m2.size_y);
  assert(m1.size_x == out.size_x);
  assert(m1.size_y == out.size_y);
  uint32_t total_size = m1.total_size();

  const float *const d1 = m1.data;
  const float *const d2 = m2.data;
  float *const o = out.data;
  for (uint32_t i = 0; i < total_size; ++i)
    o[i] = d1[i] - d2[i];
}

template <typename T, bool INVERT>
void matrix_sub_scalar(const Matrix<T> &in, T scalar, Matrix<T> &out) {
  uint32_t total_size = in.total_size();

  const float *const iptr = in.data;
  float *const optr = out.data;
  for (uint32_t i = 0; i < total_size; ++i) {
    if (INVERT) {
      optr[i] = scalar - iptr[i];
    } else {
      optr[i] = iptr[i] - scalar;
    }
  }
}

template <typename T, bool INVERT>
void matrix_sub_one(const Matrix<T> &in, Matrix<T> &out) {
  uint32_t total_size = in.total_size();

  const float *const iptr = in.data;
  float *const optr = out.data;
  for (uint32_t i = 0; i < total_size; ++i) {
    if (INVERT) {
      optr[i] = static_cast<T>(1.0f) - iptr[i];
    } else {
      optr[i] = iptr[i] - static_cast<T>(1.0f);
    }
  }
}

template <typename T>
void vector_sub(const Matrix<T> &m1, const Matrix<T> &m2, Matrix<T> &out)
{
  assert(m1.size_x == 1 || m1.size_y == 1);
  assert(m2.size_x == 1 || m2.size_y == 1);
  assert(out.size_x == 1 || out.size_y == 1);
  assert(m1.total_size() == m2.total_size());
  assert(m1.total_size() == out.total_size());

  uint32_t total_size = m1.total_size();

  const float *const d1 = m1.data;
  const float *const d2 = m2.data;
  float *const o = out.data;
  for (uint32_t i = 0; i < total_size; ++i)
    o[i] = d1[i] - d2[i];
}

template <typename T>
void matrix_ew_mult(const Matrix<T> &m1, const Matrix<T> &m2, Matrix<T> &out)
{
  assert(m1.size_x == m2.size_x);
  assert(m1.size_y == m2.size_y);
  assert(m1.size_x == out.size_x);
  assert(m1.size_y == out.size_y);
  uint32_t total_size = m1.size_x * m1.size_y;

  const float *const d1 = m1.data;
  const float *const d2 = m2.data;
  float *const o = out.data;
  for (uint32_t i = 0; i < total_size; ++i)
    o[i] = d1[i] * d2[i];
}
template <typename T> float matrix_accumulate(const Matrix<T> &m1)
{
  uint32_t total_size = m1.size_x * m1.size_y;

  const float *const d1 = m1.data;
  T accum = 0.0f;
  for (uint32_t i = 0; i < total_size; ++i)
    accum += d1[i];
  return accum;
}
template <typename T>
void matrix_mult_scalar_inplace(Matrix<T> &m, T scalar)
{
  uint32_t total_size = m.size_x * m.size_y;
  for (uint32_t i = 0; i < total_size; ++i) {
    m.data[i] *= scalar;
  }
}


template<typename T>
std::ostream &operator<<(std::ostream &stream, const Matrix<T> &matrix)
{
    stream<<matrix.size_x<<" "<<matrix.size_y;
}

template <typename T> void matrix_log(const Matrix<T> &in, Matrix<T> out) {
  uint32_t total_size = in.total_size();

  const T*const iptr = in.data;
  T *const optr = out.data;
  for (uint32_t i = 0; i < total_size; ++i)
    optr[i] = log(iptr[i]);
}

template <typename T> void matrix_log_inplace(Matrix<T> &m) {
  uint32_t total_size = m.total_size();
  T*const ptr = m.data;
  for (uint32_t i = 0; i < total_size; ++i)
    ptr[i] = log(ptr[i]);
}

} // end namespace cpu
} // end namespace core

