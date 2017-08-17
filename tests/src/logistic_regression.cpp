#include "test_utils.h"
#include <mg_ml/cpu/matrix_functions.h>
#include <mg_ml/cpu/models/logistic_regression.h>


TEST(logistic_regression, sigmoid) {

  std::vector<float> sig_data{0.0f,2.0f};
  std::vector<float> sig_data_out{0.0f,0.0f};
  core::Matrixf inm{sig_data.data(),1,2};
  core::Matrixf outm{sig_data_out.data(),1,2};
  models::cpu::sigmoid<float>(inm,outm);

  ASSERT_NEAR(outm.data[0],0.5f , 0.0000001); 
  ASSERT_NEAR(outm.data[1],0.88079708f , 0.00001); 
  //std::vector<float> A;
  //std::vector<float> Aref;
  //std::vector<float> out;
  //A.reserve(1024*32);
  //Aref.reserve(32*1024);
  //out.resize(32*1024);

  //const std::string Arefpath{"tests/A_transposed_32_1024.txt"};
  //const std::string Apath{"tests/A_1024_32.txt"};
  //load_float_mat(Apath, A);
  //load_float_mat(Arefpath, Aref);

  //EXPECT_EQ(A.size(), 1024*32);
  //EXPECT_EQ(Aref.size(), 32*1024);

  //core::Matrix<float> Am{A.data(), 1024,32};
  //core::Matrix<float> outm{out.data(), 32,1024};

  //core::cpu::matrix_transpose(Am,outm);
  //auto totalSize = 1024*32;
  //for(int i =0;i<totalSize; ++i)
  //   ASSERT_NEAR(outm.data[i], Aref[i], 0.00001); 

}
TEST(logistic_regression, simple_eval) {

    //w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
    std::vector<float> w_data{1.0f, 1.0f, 2.0f};
    std::vector<float> x_data{2.0f, 1.0f, 3.0f, 2.0f, 2.0f, 4.0f};
    std::vector<float> y_data{1.0f, 0.0f};
    std::vector<float> out_data{0.0f, 0.0f};
    core::Matrixf wm{w_data.data(), 1, 3};
    core::Matrixf xm{x_data.data(), 2, 3};
    //keeping Y transposed for easyness of computing, avoids a transpose
    core::Matrixf ym{y_data.data(), 1, 2};
    core::Matrixf outm{out_data.data(), 2, 1};

    models::cpu::simple_logistic_forward<float>(xm,wm,outm);
    ASSERT_NEAR(outm.data[0], 0.99987661 , 0.00001);
    ASSERT_NEAR(outm.data[1], 0.99999386 , 0.00001);
    float cost = models::cpu::logistic_cost<float>(outm, ym);
    ASSERT_NEAR(cost, 6.000064773192205f, 0.001);

}
