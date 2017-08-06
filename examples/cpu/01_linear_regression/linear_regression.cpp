#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <mg_ml/utils/plotting.h>

//clang++ main.cpp -std=c++11 -o gnutest && ./gnutest
using plot::plot_line;
using plot::GnuPlot;
using plot::GnuFile;

struct LineCoeff
{
    float x0;
    float x1;
};

inline float get_random_float_in_range(float min, float max) {
  return min +
         static_cast<float>(rand()) /
             (static_cast<float>(RAND_MAX / (max - min)));
}

void generate_points(float* const x , float* const y, uint32_t size)
{
    //line equation we want to gnerate points around y(x) = CONSTANT + SLOPE*x;
    const float SLOPE = 0.02;
    const float CONSTANT = 1.0f;
    const float DELTA = 15.5f;

    for (uint32_t i = 0; i < size; ++i) {
      float tempy = CONSTANT + SLOPE *static_cast<float>(i);
      float runtime_delta = get_random_float_in_range(-DELTA, DELTA);
      float min = tempy - runtime_delta;
      float max = tempy + runtime_delta;
      float finaly = min +
                 static_cast<float>(rand()) /
                     (static_cast<float>(RAND_MAX / (max - min)));
      x[i] = static_cast<float>(i);
      y[i] = finaly;
    }
}

void write_file(const float* const x, const float* const y, uint32_t size)
{
  std::ofstream myfile;
  myfile.open("example.dat");
  for (uint32_t i = 0; i < size; ++i) {
    myfile << x[i] << " " << y[i] << " \n";
  }
  myfile.close();
}


inline float cost_function(const float *const x, const float *const y,
                            uint32_t size, const LineCoeff& coeff , uint32_t feature_id)
{
    float error = 0.0f;
    if(feature_id !=0)
    {
        for( uint32_t idx =0; idx <size ; ++idx)
        {
            error += (((coeff.x0 + coeff.x1 * x[idx])  - y[idx])*x[idx]);           
        }
    }
    else
    {
        for( uint32_t idx =0; idx <size ; ++idx)
        {
            error += ((coeff.x0 + coeff.x1 * x[idx])  - y[idx]);           
        }
    
    }

    return error;
}
inline float cost_function(const float *const x, const float *const y,
                            uint32_t size, const LineCoeff& coeff)
{
    float error = 0.0f;
    for (uint32_t idx = 0; idx < size; ++idx) {
      float partial_error = ((coeff.x0 + coeff.x1 * x[idx]) - y[idx]);
      error+=  (partial_error * partial_error);
    }

    return  (1.0f / (float(size) *2.0f) ) *error;
}

LineCoeff linear_regression(const float *const x, const float *const y,
                            uint32_t size, float descent_step, float tollerance) {
  LineCoeff result{0.0f, 0.0f};
  float runtime_tollerance = 1000000.0f;
  float learning_c = descent_step * (1.0f/float(size));
  int i = 0;
  float previous = cost_function(x,y,size,result);
  std::cout<<"initial guess "<<cost_function(x,y,size,result)<<std::endl;
  while (runtime_tollerance > tollerance) {
  //for (uint32_t i = 0; i < 10000000; ++i) {
    float cost_d0 = cost_function(x, y, size, result,0);
    float cost_d1 = cost_function(x, y, size, result,1);

    result.x0 -= (learning_c* cost_d0);
    result.x1 -= (learning_c* cost_d1);
    float current_cost = cost_function(x,y,size,result);
    runtime_tollerance =  (previous - current_cost);
    if (runtime_tollerance < 0)
    {
        runtime_tollerance *= -1.0f; 
    }
    previous = current_cost;
    //previous = runtime_tollerance;
    //std::cout<<current_cost<<std::endl;
    //std::cout<<"cost at iteration "<<i<<": "<<runtime_tollerance<<std::endl;
    ++i;
  }
  std::cout<<"converge in " << i <<" iterations "<<std::endl;

  return result;
}

int main() {
  const uint32_t SIZE = 10001; 
  const float TOLLERANCE= 0.000000001f; 
  const float DESCENT_STEP= 0.00000001f; 
  float x[SIZE];
  float y[SIZE];

  generate_points(x, y, SIZE);
  write_file(x, y, SIZE);
  auto lineCoeff = linear_regression(x, y, SIZE, DESCENT_STEP, TOLLERANCE);

  GnuPlot plot;
  plot.files.resize(2);
  GnuFile &f = plot.files[0];
  f.name = "example.dat";
  std::cout<<lineCoeff.x0 << " "<<lineCoeff.x1<<std::endl;
  plot.files[1] = plot_line(lineCoeff.x0, lineCoeff.x1); 
  plot.show();

}
