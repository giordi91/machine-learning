#pragma once
#include <random>

#include <mg_ml/common/matrix.h>
#include <mg_ml/cpu/matrix_functions.h>
#include <mg_ml/cpu/models/activation_functions.h>

namespace models {
namespace cpu {

using core::Matrix;
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

#include <iostream>

void initialize_matrix_with_random_weights_0_mean_1_sd(
    Matrix<float> &W, float scaling_const, int seed = 0) {

  std::mt19937 gen(seed);
  std::normal_distribution<> d(0.0f, 1.0f);
  float* const ptr = W.data;

  uint32_t total_size = W.total_size();
  for (uint32_t i = 0; i < total_size; ++i) {
    ptr[i] = d(gen) * scaling_const;
  }
}

template <typename T>
void initialize_layers_with_random_weights(std::vector<uint32_t> &layers_size,
                                           std::vector<Matrix<T>> &layers,
                                           std::vector<T> &layers_storage, 
                                           std::vector<Matrix<T>>& biases,
                                           std::vector<T>&biases_storage) {

  // compute total layer size
  uint32_t total_size = 0;
  uint8_t total_biases_size =0;
  uint32_t layers_count = layers_size.size();
  for(uint32_t i=1; i <layers_count;++i)
  {
      total_size+= layers_size[i-1]*layers_size[i];
      total_biases_size+= layers_size[i];
  }
  //allocating memory for the layer
  layers_storage.resize(total_size);
  biases_storage.resize(total_biases_size);

  uint32_t offset=0;
  uint32_t biases_offset=0;
  T* storage_ptr = layers_storage.data(); 
  for(uint32_t i=1; i <layers_count;++i)
  {
      //generating the matrices
      uint32_t curr_offset = layers_size[i-1] *layers_size[i];
      layers.emplace_back(Matrix<T>{storage_ptr+offset, layers_size[i], layers_size[i-1]});

      biases.emplace_back(Matrix<T>{biases_storage.data() + biases_offset,1,layers_size[i]});

      offset+=curr_offset;
      biases_offset+= layers_size[i];
      initialize_matrix_with_random_weights_0_mean_1_sd(layers[i-1],0.01f);
  }
}



}//end namespace cpu
}//end namespace models
