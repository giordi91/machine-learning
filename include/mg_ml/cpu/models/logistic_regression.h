#pragma once

#include <mg_ml/cpu/models/activation_functions.h>

#include <mg_ml/common/matrix.h>

namespace models {
namespace cpu {

using core::cpu::matrix_mult_transpose;
using core::cpu::matrix_mult;
using core::cpu::matrix_log;
using core::cpu::matrix_log_inplace;
using core::cpu::matrix_sub_one;
using core::cpu::matrix_sub;
using core::cpu::matrix_transpose;
using core::cpu::matrix_mult_scalar_inplace;
using core::cpu::vector_sub;
using core::cpu::matrix_sub_in_place;

// SIMPLE LOGISTIC REGRESSION
// this is a simple logistic regression model, which assume one layer and no
// hidden layer, and makes asumption that the output Y is a vector and not a
// matrix,
// since that assumption is used for simplifying some of the computation and
// avoid
// transposition

template <typename T>
void simple_logistic_forward(const Matrix<T> &X, const Matrix<T> &W,
                             Matrix<T> &out) {
  matrix_mult_transpose<T>(X, W, out);
  sigmoid_inplace<T>(out);
}

template <typename T>
void simple_logistic_backwards(const Matrix<T> &X, const Matrix<T> &A,
                               const Matrix<T> &Y, Matrix<T> &grad) {

  std::vector<T> A_minus_Y(Y.total_size());
  Matrix<T> A_minus_Ym{A_minus_Y.data(), Y.size_x, Y.size_y};
  // here we exploit the fact that A and Y are vectors, so doesnt matter if they
  // are
  // colum or row, we access data in the way we need it so we perform the sub,
  // in doing so, then the vector is ready to be multiplied by X
  vector_sub(A, Y, A_minus_Ym);

  // possible transpose here happening
  matrix_mult<T>(A_minus_Ym, X, grad);
  matrix_mult_scalar_inplace(grad, (1.0f / float(X.size_x)));
}

template <typename T>
void simple_logistic_apply_grad(Matrix<T> &W, Matrix<T> grad,
                                float learing_rate) {
  matrix_mult_scalar_inplace(grad, learing_rate);
  matrix_sub_in_place(W, grad);
}

template <typename T> T logistic_cost(const Matrix<T> &in, const Matrix<T> &Y) {
  // computing log of the matrix
  std::vector<T> log_data(in.total_size());
  Matrix<T> logM{log_data.data(), in.size_x, in.size_y};
  matrix_log(in, logM);

  // computing first part y*log(a); meaning the expected result time the log of
  // activation
  T out1;
  Matrix<T> out1M{&out1, 1, 1};
  matrix_mult<T>(Y, logM, out1M);

  // computing second part
  //(1-Y) * log(1-a)
  matrix_sub_one<T, true>(in, logM);
  matrix_log_inplace(logM);
  std::vector<T> y_sub_data(Y.total_size());
  Matrix<T> y_sub_m{y_sub_data.data(), Y.size_x, Y.size_y};
  matrix_sub_one<T, true>(Y, y_sub_m);
  T out2;
  Matrix<T> out2M{&out2, 1, 1};
  matrix_mult<T>(y_sub_m, logM, out2M);
  // ready to return
  return (-static_cast<T>(1.0f) / in.size_x) * (out1 + out2);
}

} // end namespace cpu
} // end namespace models
