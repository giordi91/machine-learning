#include <mg_ml/cpu/matrix.h>

#include <iostream>
#include <cassert>

namespace core {
namespace cpu {

void mult_matrix( const Matrix& m1, const Matrix& m2, Matrix& out)
{
    assert(m1.size_y == m2.size_x);
    assert(m1.size_x == out.size_x);
    assert(m1.size_y == out.size_y);
    std::cout<<"LOL Matrix mult"<<std::endl;

}
void matrix_mult_transpose( const Matrix& m1, const Matrix& m2, Matrix& out)
{
    assert(m1.size_y == m2.size_y);
    assert(m1.size_x == out.size_x);
    assert(m2.size_x == out.size_y);
    
    const float* const d1 = m1.data;
    const float* const d2 = m2.data;
    float* const o = out.data;

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

void matrix_sub( const Matrix& m1, const Matrix& m2, Matrix& out)
{
    //std::cout<<"size m1 "<<m1.size_x<<" "<<m1.size_y<<std::endl;
    //std::cout<<"size m2 "<<m2.size_x<<" "<<m2.size_y<<std::endl;
    //std::cout<<"size out "<<out.size_x<<" "<<out.size_y<<std::endl;
    assert(m1.size_x == m2.size_x);
    assert(m1.size_y == m2.size_y);
    assert(m1.size_x == out.size_x);
    assert(m1.size_y == out.size_y);
    uint32_t total_size = m1.size_x * m1.size_y;

    const float* const d1 = m1.data;
    const float* const d2 = m2.data;
    float* const o = out.data;
    for(uint32_t i =0; i<total_size; ++i)
        o[i] = d1[i] -d2[i];    

}

void matrix_ew_mult( const Matrix& m1, const Matrix& m2, Matrix& out)
{
    //std::cout<<"size m1 "<<m1.size_x<<" "<<m1.size_y<<std::endl;
    //std::cout<<"size m2 "<<m2.size_x<<" "<<m2.size_y<<std::endl;
    //std::cout<<"size out "<<out.size_x<<" "<<out.size_y<<std::endl;
    assert(m1.size_x == m2.size_x);
    assert(m1.size_y == m2.size_y);
    assert(m1.size_x == out.size_x);
    assert(m1.size_y == out.size_y);
    uint32_t total_size = m1.size_x * m1.size_y;

    const float* const d1 = m1.data;
    const float* const d2 = m2.data;
    float* const o = out.data;
    for(uint32_t i =0; i<total_size; ++i)
        o[i] = d1[i] * d2[i];    

}

float matrix_accumulate( const Matrix& m1)
{
    uint32_t total_size = m1.size_x * m1.size_y;

    const float* const d1 = m1.data;
    float accum =0.0f;
    for(uint32_t i =0; i<total_size; ++i)
        accum += d1[i];    
    return accum;


}

} // end namespace cpu
} // end namespace core

