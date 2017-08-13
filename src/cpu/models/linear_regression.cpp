#include <iostream>
#include <memory>
#include <mg_ml/cpu/models/linear_regression.h>
#include <vector>

namespace core {
namespace cpu {

inline void cost_function_grad(const Matrix &mx, const Matrix &my,
                               const Matrix &coeff, Matrix outCoeff,
                               Matrix &intermediate) {

  // here we compute the regular cost by computing the weights time the samples
  matrix_mult_transpose( coeff,mx, intermediate);
  // we subtract the error so we compute the error
  matrix_sub(intermediate, my, intermediate);

  //here we allocate temporary data to get the transpose of the intremediate matrix 
  std::vector<float> transpose_intermediate;
  transpose_intermediate.resize(intermediate.size_y * intermediate.size_x);
  
  //in order to get the right matrix size we need to transpose the intermediate size and
  //perform a matrix multiplication, this is not the most efficient way to do it, if 
  //we can transpose my, we can get away with one transpose only of mx 
  Matrix transp{transpose_intermediate.data(), intermediate.size_y, intermediate.size_x};
  matrix_transpose(intermediate, transp);
  matrix_mult(transp, mx, outCoeff);
}

inline void cost_function_grad_y_transposed(const Matrix &mx, const Matrix &my,
                               const Matrix &coeff, Matrix outCoeff,
                               Matrix &intermediate) {

  // here we compute the regular cost by computing the weights time the samples
  matrix_mult_transpose( coeff,mx, intermediate);
  // we subtract the error so we compute the error
  vector_sub(intermediate, my, intermediate);
  
  //multiplying the resulting cost by the colum feature of x
  matrix_mult(intermediate, mx, outCoeff);
}

inline float cost_function(const Matrix &mx, const Matrix &my,
                           Matrix &intermediate, uint32_t size,
                           const Matrix &coeff) {
  // here we multiply the weights by the samples
  matrix_mult_transpose(mx, coeff, intermediate);
  // we subtract the error
  matrix_sub(intermediate, my, intermediate);
  // finally we accumulate the error vector
  matrix_ew_mult(intermediate, intermediate, intermediate);
  // final error is devided by twice the size of the vector
  return (1.0f / (float(size) * 2.0f)) * matrix_accumulate(intermediate);
}

void linear_regression(const Matrix &mx, const Matrix &my, uint32_t size,
                       float learning_rate, Matrix &result,
                       uint32_t iterations) {
  float learning_c = learning_rate * (1.0f / float(size));
  auto temp_m = std::make_unique<float[]>(mx.size_x * result.size_x);
  Matrix temp{temp_m.get(), result.size_x, mx.size_x };
  std::vector<float> outRes;
  outRes.resize(result.size_x * result.size_y);
  Matrix tempRes{outRes.data(), result.size_x, result.size_y};

  for (uint32_t i = 0; i < iterations; ++i) {
    cost_function_grad_y_transposed(mx, my, result, tempRes, temp);

    // mult_by_learning_rate
    matrix_mult_scalar_inplace(tempRes, learning_c);
    // update final
    matrix_sub(result, tempRes, result);
  }
}
} // end namespace cpu
} // end namespace core
