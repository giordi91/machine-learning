#include "test_utils.h"
#include <mg_ml/cpu/matrix_functions.h>
#include <mg_ml/cpu/models/logistic_regression.h>

TEST(random_init, init_in_range) {

  std::vector<float> data(2000);
  core::Matrix<float> inm = {data.data(), 2000,1};

  core::cpu::initialize_to_rand_in_range<float>(inm, -0.5f, -0.5f );
  const float* const  ptr = data.data();
  for (int i = 0; i < 2000; ++i) {
    ASSERT_LE(ptr[i], 0.5f);
    ASSERT_GE(ptr[i], -0.5f);
  }
  //ASSERT_NEAR(outm.data[0], 0.5f, 0.0000001);
  //ASSERT_NEAR(outm.data[1], 0.88079708f, 0.00001);
}
