#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include <mg_ml/cpu/matrix.h>

// faking some test file input; behaves like an `std::ifstream`:
//std::istringstream is("1,2,3,4,5");
//
//std::vector<int> numbers;
//
//std::string number_as_string;
//while (std::getline(is, number_as_string, ','))
//{
//    numbers.push_back(std::stoi(number_as_string));
//}

TEST(matrix_mult_transpose, naive) {

  std::ifstream file("tests/A_1024_32.txt");
  std::vector<float> numbers;

  if (file) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    std::string number_as_string;
    while (std::getline(buffer, number_as_string, ','))
        numbers.push_back(std::stof(number_as_string));
  }
  else
  {
      std::cout<<"NOOOOOOOOOOOOO"<<std::endl;
  }

  EXPECT_EQ(1, 1);
}
