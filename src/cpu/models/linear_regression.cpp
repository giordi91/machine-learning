#include<mg_ml/cpu/models/linear_regression.h>
#include <iostream>
#include <memory>

namespace core {
namespace cpu {

inline float cost_function_grad(const Matrix &mx, const Matrix &my,
                                uint32_t size, const Matrix &coeff,
                                uint32_t feature_id, Matrix &intermediate) {
  const float *const x = mx.data;
  const float *const y = my.data;
  float error = 0.0f;
  if (feature_id != 0) {
    for (uint32_t idx = 0; idx < size; ++idx) {
      error += (((coeff.data[0] + coeff.data[1] * x[idx * 2 + 1]) - y[idx]) *
                x[idx * 2 + 1]);
    }
    matrix_mult_transpose(mx,coeff,intermediate);
    //we subtract the error
    matrix_sub(intermediate,my,intermediate);
  } else {
    for (uint32_t idx = 0; idx < size; ++idx) {
      error += ((coeff.data[0] + coeff.data[1] * x[idx * 2 + 1]) - y[idx]);
    }
  }
  return error;
}
inline float cost_function(const Matrix& mx,  const Matrix& my, Matrix& intermediate,
                            uint32_t size, const Matrix& coeff)
{
    //here we multiply the weights by the samples
    matrix_mult_transpose(mx,coeff,intermediate);
    //we subtract the error
    matrix_sub(intermediate,my,intermediate);
    //finally we accumulate the error vector
    matrix_ew_mult(intermediate,intermediate, intermediate);
    //final error is devided by twice the size of the vector
    return  (1.0f / (float(size) *2.0f) )* matrix_accumulate(intermediate);
}

void linear_regression(const Matrix &mx, const Matrix &my, uint32_t size,
                       float learning_rate, Matrix &result, uint32_t iterations)
{
  //LineCoeff result{0.0f, 0.0f};
 float learning_c = learning_rate * (1.0f / float(size));
 auto temp_m = std::make_unique<float[]>(mx.size_x * result.size_x);
 Matrix temp{temp_m.get(), mx.size_x, result.size_x};
 for (uint32_t i = 0; i < iterations; ++i) {
   float cost_d0 = cost_function_grad(mx, my, size, result, 0, temp);
   float cost_d1 = cost_function_grad(mx, my, size, result, 1, temp);

   result.data[0] -= (learning_c * cost_d0);
   result.data[1] -= (learning_c * cost_d1);
   //float current_cost = cost_function(mx, my, temp, size, result);

  }
}
} // end namespace cpu
} // end namespace core
