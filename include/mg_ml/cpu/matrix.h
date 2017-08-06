#pragma once
#include <cstdint>

namespace core {
namespace cpu {

struct Matrix {
  float *data = nullptr;
  uint32_t size_x = 0;
  uint32_t size_y = 0;
};

void mult_matrix( const Matrix& m1, const Matrix& m2, Matrix& out);
void matrix_mult_transpose( const Matrix& m1, const Matrix& m2, Matrix& out);
void matrix_sub( const Matrix& m1, const Matrix& m2, Matrix& out);
void matrix_ew_mult( const Matrix& m1, const Matrix& m2, Matrix& out);
float matrix_accumulate( const Matrix& m1);

}//namespace cpu 
}//namespcae core

