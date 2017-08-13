#include "test_utils.h"
#include <mg_ml/cpu/matrix_functions.h>


TEST(matrix_transpose, naive) {

  std::vector<float> A;
  std::vector<float> Aref;
  std::vector<float> out;
  A.reserve(1024*32);
  Aref.reserve(32*1024);
  out.resize(32*1024);

  const std::string Arefpath{"tests/A_transposed_32_1024.txt"};
  const std::string Apath{"tests/A_1024_32.txt"};
  load_float_mat(Apath, A);
  load_float_mat(Arefpath, Aref);

  EXPECT_EQ(A.size(), 1024*32);
  EXPECT_EQ(Aref.size(), 32*1024);

  core::Matrix Am{A.data(), 1024,32};
  core::Matrix outm{out.data(), 32,1024};

  core::cpu::matrix_transpose(Am,outm);
  auto totalSize = 1024*32;
  for(int i =0;i<totalSize; ++i)
     ASSERT_NEAR(outm.data[i], Aref[i], 0.00001); 

}
