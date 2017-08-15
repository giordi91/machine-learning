#pragma once
#include <cstdint>
namespace core {

template<typename T>
struct Matrix {
  T* data = nullptr;
  uint32_t size_x = 0;
  uint32_t size_y = 0;

  inline uint32_t total_size() const { return size_x * size_y; }
};
} // end namespace core
