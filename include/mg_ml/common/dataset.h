#pragma once
#include <vector>
#include <string>

#include <mg_ml/common/matrix.h>

namespace dataset {
using core::MatrixI;

enum class CFarClasses
{
    AIRPLANE=0,
    AUTOMOBILE,
    BIRD,
    CAT,
    DEER,
    DOG,
    FROG,
    HORSE,
    SHIP,
    TRUCK
}

bool load_cifar_10(const std::string &rootpath, MatrixI<uint8_t> &X, MatrixI<uint8_t> &Y,
                   std::vector<uint8_t> &Xstorage, std::vector<uint8_t> &Ystorage);
bool dump_image_from_cifar_10_dataset(const std::string &outpath,
                                      MatrixI<uint8_t> &data, uint32_t index);

} // end namespace dataset
