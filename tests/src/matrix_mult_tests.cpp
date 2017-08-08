#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include <mg_ml/cpu/matrix.h>

void load_float_mat(const std::string& path, std::vector<float>& data)
{
  std::ifstream file(path);
  if (file) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    std::string number_as_string;
    while (std::getline(buffer, number_as_string))
    {
        data.push_back(std::stof(number_as_string));
    }
  }
  else
  {
      EXPECT_FALSE(true) << "ERROR Test file not found: "<<path;
  }

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

  core::cpu::Matrix Am{A.data(), 1024,32};
  core::cpu::Matrix Cm{C.data(), 1024,32};
  core::cpu::Matrix outm{out.data(), 1024,1024};

  core::cpu::matrix_mult_transpose(Am,Cm, outm);
  auto totalSize = 1024*1024;
  for(int i =0;i<totalSize; ++i)
     ASSERT_NEAR(outm.data[i], ref[i], 0.00001); 

}
