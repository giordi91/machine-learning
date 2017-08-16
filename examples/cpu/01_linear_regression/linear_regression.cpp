#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

#include <mg_ml/common/plotting.h>
#include <mg_ml/cpu/matrix_functions.h>
#include <mg_ml/cpu/models/linear_regression.h>

// clang++ main.cpp -std=c++11 -o gnutest && ./gnutest
using plot::plot_line;
using plot::GnuPlot;
using plot::GnuFile;
using core::Matrix;

void generate_points(Matrix<float> *mx, Matrix<float> *my, uint32_t size) {
  // line equation we want to gnerate points around y(x) = CONSTANT + SLOPE*x;
  const float SLOPE = 0.02;
  const float CONSTANT = 1.0f;
  const float DELTA = 15.5f;

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(-DELTA, DELTA);
  for (uint32_t i = 0; i < size; ++i) {
    float tempy = CONSTANT + SLOPE * static_cast<float>(i);

    float runtime_delta = dist(mt);
    float min = tempy - runtime_delta;
    float max = tempy + runtime_delta;
    std::uniform_real_distribution<float> dist2(min, max);
    float finaly =
        min + dist2(mt) / (static_cast<float>(RAND_MAX / (max - min)));
    mx->data[i * 2] = 1.0f;
    mx->data[i * 2 + 1] = static_cast<float>(i);
    my->data[i] = finaly;
  }
}

void write_file(const float *const x, const float *const y, uint32_t size) {
  std::ofstream myfile;
  myfile.open("example.dat");
  for (uint32_t i = 0; i < size; ++i) {
    myfile << x[i * 2 + 1] << " " << y[i] << " \n";
  }
  myfile.close();
}

int main() {
  const uint32_t SIZE = 10001;
  const float DESCENT_STEP = 0.00000001f;

  auto xmem = std::make_unique<float[]>(SIZE * 2);
  auto ymem = std::make_unique<float[]>(SIZE);
  Matrix<float> mx2{xmem.get(), SIZE, 2};
  Matrix<float> my2{ymem.get(), SIZE, 1};

  // generate_points(mx.data, my.data, SIZE);
  generate_points(&mx2, &my2, SIZE);
  write_file(mx2.data, my2.data, SIZE);

  float coeffs[2] = {0.0f, 0.0f};
  Matrix<float> coeffM{coeffs, 1, 2};
  core::cpu::linear_regression(mx2, my2, SIZE, DESCENT_STEP, coeffM, 200);

  GnuPlot plot;
  plot.files.resize(2);
  GnuFile &f = plot.files[0];
  f.name = "example.dat";
  plot.files[1] = plot_line(coeffM.data[0], coeffM.data[1]);
  plot.show();
}
