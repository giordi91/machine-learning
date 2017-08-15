#pragma once
#include <cstdint>
#include <iostream>
#include <vector>
#include <cassert>

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
  uint32_t total_size = m1.size_x * m1.size_y;

  const float *const d1 = m1.data;
  const float *const d2 = m2.data;
  float *const o = out.data;
  for (uint32_t i = 0; i < total_size; ++i)
    o[i] = d1[i] - d2[i];
}
template <typename T>
void vector_sub(const Matrix<T> &m1, const Matrix<T> &m2, Matrix<T> &out)
{
  assert(m1.size_x == 1 || m1.size_y == 1);
  assert(m2.size_x == 1 || m2.size_y == 1);
  assert(out.size_x == 1 || out.size_y == 1);
  assert(m1.total_size() == m2.total_size());
  assert(m1.total_size() == out.total_size());

  uint32_t total_size = m1.size_x * m1.size_y;

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
  float accum = 0.0f;
  for (uint32_t i = 0; i < total_size; ++i)
    accum += d1[i];
  return accum;
}
template <typename T>
void matrix_mult_scalar_inplace(Matrix<T> &m, float scalar)
{
  uint32_t total_size = m.size_x * m.size_y;
  for (uint32_t i = 0; i < total_size; ++i) {
    m.data[i] *= scalar;
  }
}
// std::ostream &operator<<(std::ostream &stream, const Matrix<T> &matrix);


} // end namespace cpu
} // end namespace core

