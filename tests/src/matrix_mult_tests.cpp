#include "test_utils.h"
#include <mg_ml/cpu/matrix_functions.h>

TEST(matrix_mult, naive) {

  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> ref;
  std::vector<float> out;
  A.reserve(1024*32);
  B.reserve(32*64);
  out.reserve(1024*64);
  ref.reserve(1024*64);

  const std::string Apath{"tests/A_1024_32.txt"};
  const std::string Bpath{"tests/B_32_64.txt"};
  const std::string refpath{"tests/AxB_1024_64.txt"};
  load_float_mat(Apath, A);
  load_float_mat(Bpath, B);
  load_float_mat(refpath, ref);

  EXPECT_EQ(A.size(), 1024*32);
  EXPECT_EQ(B.size(), 32*64);
  EXPECT_EQ(ref.size(), 1024*64);

  core::Matrix<float> Am{A.data(), 1024,32};
  core::Matrix<float> Bm{B.data(), 32,64};
  core::Matrix<float> outm{out.data(), 1024,64};

  core::cpu::matrix_mult<float>(Am,Bm, outm);
  auto totalSize = 1024*64;
  for(int i =0;i<totalSize; ++i)
     ASSERT_NEAR(outm.data[i], ref[i], 0.00001); 

}

TEST(matrix_mult_transpose, naive) {
  std::vector<float> A;
  std::vector<float> C;
  std::vector<float> ref;
  std::vector<float> out;
  A.reserve(1024*32);
  C.reserve(1024*32);
  out.reserve(1024*1024);
  ref.reserve(1024*1024);

  const std::string Apath{"tests/A_1024_32.txt"};
  const std::string Cpath{"tests/C_1024_32.txt"};
  const std::string refpath{"tests/AxC_transposed_1024_1024.txt"};
  load_float_mat(Apath, A);
  load_float_mat(Cpath, C);
  load_float_mat(refpath, ref);

  EXPECT_EQ(A.size(), 1024*32);
  EXPECT_EQ(C.size(), 1024*32);
  EXPECT_EQ(ref.size(), 1024*1024);

  core::Matrix<float> Am{A.data(), 1024,32};
  core::Matrix<float> Cm{C.data(), 1024,32};
  core::Matrix<float> outm{out.data(), 1024,1024};

  core::cpu::matrix_mult_transpose<float>(Am,Cm, outm);
  auto totalSize = 1024*1024;
  for(int i =0;i<totalSize; ++i)
     ASSERT_NEAR(outm.data[i], ref[i], 0.00001); 

}
