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

}//namespace cpu 
}//namespcae core

