#include <mg_ml/common/dataset.h>
#include <fstream>
#include <iostream>
#include <sstream>
namespace dataset {

bool load_cifar_10(const std::string &rootpath, MatrixI<char> &X, MatrixI<char> &Y,
                   std::vector<char> &Xstorage, std::vector<char> &Ystorage)
  //the cifar_10 is composed of of 5 different files, lests loop them
{
  const uint32_t IMAGE_DATA_SIZE = 3072u;
  const uint32_t IMAGES_PER_FILE = 10000u;
  const uint32_t NUMBER_OF_FILES = 5u;
  Xstorage.resize(IMAGE_DATA_SIZE * IMAGES_PER_FILE* NUMBER_OF_FILES);
  Ystorage.resize( IMAGES_PER_FILE * NUMBER_OF_FILES);

  char* const yptr = Ystorage.data();
  char* const xptr = Xstorage.data();
  
  for (uint32_t i = 0; i < 5; ++i) {


    std::string path = rootpath + "data_batch_" + std::to_string(i+1) + ".bin"; 
    std::cout<<"opening file "<<path<<std::endl;

    std::ifstream file (path, std::ios::in | std::ios::binary);
    if (!file) {
      Xstorage.clear();
      Ystorage.clear();
      return false;
    }
    for (uint32_t f =0; f<10000;++f)
    {
        //reading the class of the image 
        char* currClass = yptr + i*IMAGES_PER_FILE + f;
        file.read(currClass, 1);
        
        //reading the image
        char *currImage = xptr + (i * IMAGES_PER_FILE + f) * IMAGE_DATA_SIZE;
        file.read(currImage, IMAGE_DATA_SIZE);
    }
  }

  X.data = Xstorage.data();
  Y.data = Ystorage.data();
  
  X.size_x =  NUMBER_OF_FILES* IMAGES_PER_FILE;
  X.size_y =  IMAGE_DATA_SIZE;

  Y.size_x = 1;
  Y.size_y = NUMBER_OF_FILES* IMAGES_PER_FILE;

  return true;
}
}// end namespace dataset
