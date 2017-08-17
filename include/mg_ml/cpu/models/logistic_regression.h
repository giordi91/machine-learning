#pragma once

#include <mg_ml/cpu/models/activation_functions.h>

#include <mg_ml/common/matrix.h>


namespace models {
namespace cpu {

using core::cpu::matrix_mult_transpose;

template<typename T>
void simple_logistic_forward( const Matrix<T>& X, const Matrix<T>& W, Matrix<T>& out)
{
    core::cpu::matrix_mult_transpose<T>(X,W,out);
    sigmoid_inplace<T>(out);
}

template<typename T>
void simple_logistic_backwards( const Matrix<T>& X, const Matrix<T>& W, Matrix<T>& out)
{
}

template<typename T>
T logistic_cost( const Matrix<T>& in)
{
    return 0.0f;

}
} // end namespace cpu
} // end namespace models 
