#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

#include <mg_ml/cpu/matrix.h>
#include <mg_ml/utils/plotting.h>

//clang++ main.cpp -std=c++11 -o gnutest && ./gnutest
using plot::plot_line;
using plot::GnuPlot;
using plot::GnuFile;
using core::cpu::Matrix;


//struct LineCoeff
//{
//    float x0;
//    float x1;
//};

void generate_points(Matrix& mx , Matrix& my, uint32_t size)
{
    //line equation we want to gnerate points around y(x) = CONSTANT + SLOPE*x;
    const float SLOPE = 0.02;
    const float CONSTANT = 1.0f;
    const float DELTA = 15.5f;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-DELTA, DELTA);
    for (uint32_t i = 0; i < size; ++i) {
      float tempy = CONSTANT + SLOPE *static_cast<float>(i);

      float runtime_delta = dist(mt);
      float min = tempy - runtime_delta;
      float max = tempy + runtime_delta;
      std::uniform_real_distribution<float> dist2(min, max);
      float finaly = min + dist2(mt) / (static_cast<float>(RAND_MAX / (max - min)));
      mx.data[i*2   ] = 1.0f;
      mx.data[i*2 +1] = static_cast<float>(i);
      my.data[i] = finaly;
    }
}

void write_file(const float* const x, const float* const y, uint32_t size)
{
  std::ofstream myfile;
  myfile.open("example.dat");
  for (uint32_t i = 0; i < size; ++i) {
    myfile << x[i*2+1] << " " << y[i] << " \n";
  }
  myfile.close();
}


inline float cost_function_grad(const Matrix& mx,  const Matrix& my,
                            uint32_t size, const Matrix& coeff , uint32_t feature_id)
{
    const float *const x = mx.data;
    const float *const y = my.data;
    float error = 0.0f;
    if(feature_id !=0)
    {
        for( uint32_t idx =0; idx <size ; ++idx)
        {
            error += (((coeff.data[0] + coeff.data[1] * x[idx*2 +1])  - y[idx])*x[idx*2+1]);           
        }
    }
    else
    {
        for( uint32_t idx =0; idx <size ; ++idx)
        {
            error += ((coeff.data[0] + coeff.data[1] * x[idx*2+1])  - y[idx]);           
        }
    
    }

    return error;
}
inline float cost_function(const Matrix& mx,  const Matrix& my, Matrix& intermediate,
                            uint32_t size, const Matrix& coeff)
{
    //here we multiply the weights by the smaples
    matrix_mult_transpose(mx,coeff,intermediate);
    //we subtract the error
    matrix_sub(intermediate,my,intermediate);
    //finally we accumulate the error vector
    matrix_ew_mult(intermediate,intermediate, intermediate);
    //final error is devided by twice the size of the vector
    return  (1.0f / (float(size) *2.0f) )* matrix_accumulate(intermediate);
    //const float *const x = mx.data;
    //const float *const y = my.data;
    //float error = 0.0f;
    //for (uint32_t idx = 0; idx < size; ++idx) {
    //  float partial_error = ((coeff.data[0] + coeff.data[1] * x[idx*2+1]) - y[idx]);
    //  error+=  (partial_error * partial_error);
    //}
    //return  
}

void linear_regression(const Matrix& mx,  const Matrix& my,
                            uint32_t size, float descent_step, float tollerance, Matrix& result) {
  //LineCoeff result{0.0f, 0.0f};
  float runtime_tollerance = 1000000.0f;
  float learning_c = descent_step * (1.0f/float(size));
  int i = 0;
  auto temp_m = std::make_unique<float[]>(mx.size_x*result.size_x);
  //std::cout<<"intermediate size "<< mx.size_x*result.size_x<<std::endl;
  Matrix temp{temp_m.get(), mx.size_x, result.size_x};
  float previous = cost_function(mx,my,temp,size,result);
  //while (runtime_tollerance > tollerance) {
  for (uint32_t i = 0; i < 1000; ++i) {
  float cost_d0 = cost_function_grad(mx, my, size, result, 0);
  float cost_d1 = cost_function_grad(mx, my, size, result, 1);

  result.data[0] -= (learning_c * cost_d0);
  result.data[1] -= (learning_c * cost_d1);
  float current_cost = cost_function(mx, my, temp, size, result);
  runtime_tollerance = (previous - current_cost);
  if (runtime_tollerance < 0) {
    runtime_tollerance *= -1.0f; 
    }
    previous = current_cost;
    
  }
  std::cout<<"converge in " << i <<" iterations "<<std::endl;

}

int main() {
  const uint32_t SIZE = 10001; 
  const float TOLLERANCE= 0.000000001f; 
  const float DESCENT_STEP= 0.00000001f; 
  
  auto xmem = std::make_unique<float[]>(SIZE*2);
  auto ymem = std::make_unique<float[]>(SIZE);
  Matrix mx2 {xmem.get(),SIZE,2};
  Matrix my2 {ymem.get(),SIZE,1};

  //generate_points(mx.data, my.data, SIZE);
  generate_points(mx2, my2, SIZE);
  write_file(mx2.data, my2.data, SIZE);

  float coeffs[2]={0.0f,0.0f};
  Matrix coeffM{coeffs, 1,2};
  linear_regression(mx2, my2, SIZE, DESCENT_STEP, TOLLERANCE,coeffM);

  GnuPlot plot;
  plot.files.resize(2);
  GnuFile &f = plot.files[0];
  f.name = "example.dat";
  std::cout<<coeffM.data[0] << " "<<coeffM.data[1]<<std::endl;
  plot.files[1] = plot_line(coeffM.data[0], coeffM.data[1]); 
  plot.show();
}
