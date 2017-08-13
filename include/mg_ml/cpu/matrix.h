#pragma once
#include <cstdint>
#include <iostream>

namespace core {
namespace cpu {

struct Matrix {
  float *data = nullptr;
  uint32_t size_x = 0;
  uint32_t size_y = 0;

  inline uint32_t total_size()const
  { return size_x*size_y;}
};

void matrix_mult(const Matrix &m1, const Matrix &m2, Matrix &out);
void matrix_transpose(const Matrix &m, Matrix &out);
void matrix_mult_transpose(const Matrix &m1, const Matrix &m2, Matrix &out);
void matrix_sub(const Matrix &m1, const Matrix &m2, Matrix &out);
void vector_sub(const Matrix &m1, const Matrix &m2, Matrix &out);
void matrix_ew_mult(const Matrix &m1, const Matrix &m2, Matrix &out);
float matrix_accumulate(const Matrix &m1);
void matrix_mult_scalar_inplace(Matrix &m, float scalar);

std::ostream& operator<< (std::ostream& stream, const Matrix& matrix); 

}//namespace cpu 
}//namespcae core

