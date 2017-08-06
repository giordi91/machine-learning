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

} // end namespace cpu
} // end namespace core

