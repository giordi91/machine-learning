#pragma once
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

inline void load_float_mat(const std::string& path, std::vector<float>& data)
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

