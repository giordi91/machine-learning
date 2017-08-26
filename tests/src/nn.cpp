#include "test_utils.h"
#include <mg_ml/cpu/matrix_functions.h>
#include <mg_ml/cpu/models/neural_network.h>

TEST(nn, naive) {

    //3x3
  std::vector<float> Xstorage {  1.62434536f , -1.07296862f,
                                -0.61175641f ,  0.86540763f,
                                -0.52817175f , -2.3015387f};

  //1x3
  std::vector<float> Ystorage{1.62434536f, -0.61175641f, -0.52817175f};

  //4x3
  std::vector<float> W1 {-0.00416758f ,-0.00056267f,
                         -0.02136196f , 0.01640271f,
                         -0.01793436f ,-0.00841747f,
                          0.00502881f ,-0.01245288f};

  std::vector<float> b1 {0.0f,0.0f,0.0f,0.0f};
                         
                         
  std::vector<float> W2{-0.01057952f, -0.00909008f, 0.00551454f, 0.02292208f};
  std::vector<float> b2 {0.0f};

  core::Matrix<float> X{Xstorage.data(), 3, 2};
  core::Matrix<float> Y{Xstorage.data(), 1, 2};

  std::vector<uint32_t>layer_size{2, 4, 1};
  std::vector<float>W_storage; 
  std::vector<float>biases_storage; 
  std::vector<core::Matrix<float>> W_layers;
  std::vector<core::Matrix<float>> biases;
  
  //initializing layers
  models::cpu::initialize_layers_with_random_weights<float>(
      layer_size, W_layers, W_storage, biases, biases_storage);

  //assert total allocation
  ASSERT_EQ(W_storage.size(),12 );
  
  //asserting matrices sizes
  ASSERT_EQ(W_layers[0].size_x,4);
  ASSERT_EQ(W_layers[0].size_y,2);
  
  ASSERT_EQ(W_layers[1].size_x,1);
  ASSERT_EQ(W_layers[1].size_y,4);

  //asserting pointers
  ASSERT_EQ(W_storage.data(), W_layers[0].data);
  //shifting by 20 since the first matrix is a 4x5
  ASSERT_EQ(W_storage.data()+8, W_layers[1].data);

  //checking biases
  ASSERT_EQ(biases_storage.size(),5 );
  ASSERT_EQ(biases[0].size_x,1);
  ASSERT_EQ(biases[1].size_x,1);

  ASSERT_EQ(biases[0].size_y,layer_size[1]);
  ASSERT_EQ(biases[1].size_y,layer_size[2]);

  ASSERT_EQ(biases[0].data,biases_storage.data());
  ASSERT_EQ(biases[1].data,biases_storage.data()+4);

}
