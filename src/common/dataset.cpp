#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include <mg_ml/common/dataset.h>

namespace dataset {

bool load_cifar_10(const std::string &rootpath, Matrix<uint8_t> &X, Matrix<uint8_t> &Y,
                   std::vector<uint8_t> &Xstorage, std::vector<uint8_t> &Ystorage)
  //the cifar_10 is composed of of 5 different files, lests loop them
{
  const uint32_t IMAGE_DATA_SIZE = 3072u;
  const uint32_t IMAGES_PER_FILE = 10000u;
  const uint32_t NUMBER_OF_FILES = 5u;
  Xstorage.resize(IMAGE_DATA_SIZE * IMAGES_PER_FILE* NUMBER_OF_FILES);
  Ystorage.resize( IMAGES_PER_FILE * NUMBER_OF_FILES);

  uint8_t* const yptr = Ystorage.data();
  uint8_t* const xptr = Xstorage.data();
  
  for (uint32_t i = 0; i < 5; ++i) {


    std::string path = rootpath + "data_batch_" + std::to_string(i+1) + ".bin"; 
    std::cout<<"opening file "<<path<<std::endl;

    std::ifstream file (path, std::ios::in | std::ios::binary);
    if (!file) {
      Xstorage.clear();
      Ystorage.clear();
      return false;
    }
    //each batch file contains 10k images
    for (uint32_t f =0; f<10000;++f)
    {
        //reading the class of the image 

        uint8_t* currClass = yptr + i*IMAGES_PER_FILE + f;
        file.read(reinterpret_cast<char*>(currClass), 1);
        
        //reading the image
        uint8_t *currImage = xptr + (i * IMAGES_PER_FILE + f) * IMAGE_DATA_SIZE;
        file.read(reinterpret_cast<char*>(currImage), IMAGE_DATA_SIZE);
    }
  }

  X.data = Xstorage.data();
  Y.data = Ystorage.data();
  
  X.size_x =  NUMBER_OF_FILES* IMAGES_PER_FILE;
  X.size_y =  IMAGE_DATA_SIZE;

  //hardcoding row vector of values
  Y.size_x = 1;
  Y.size_y = NUMBER_OF_FILES* IMAGES_PER_FILE;

  return true;
}

//assum each row is a picture 
bool dump_image_from_cifar_10_dataset(const std::string &outpath,
                                      Matrix<uint8_t> &data, uint32_t index)
{
    
    std::ofstream out_file;
    out_file.open(outpath, std::ios::out);
    if (!out_file) {
      return false;
    }
    //
    //size of the picture is the colum count of the matrix, since each row is a full
    //picture, the data is first all the r, then all the g then all the b, so if 
    //we divide by 3 we get the total number of pixels, taking squre root since 
    //image from cifar 10 is squared
    uint32_t pic_size = sqrt(std::floor((data.size_y/3.0))); 
    uint32_t pic_size_sq = pic_size * pic_size ;
    //shifting the pointer of the image by how many images we wish to skip
    const uint8_t* ptr= data.data + data.size_y*index;

    for (uint32_t r = 0; r < pic_size; ++r) {
      for (uint32_t c = 0; c < pic_size; ++c) {
          //adding pixel coordinate
          //coordinate of the pixel, needed to do some permutation in order 
          //for the picture to show oriented correctly
          out_file<<std::to_string(c)<<" "<<std::to_string(32-r)<<" ";
          //writing R color
          uint32_t pix_pos = r*pic_size+c;
          out_file<<std::to_string(ptr[pix_pos])<<" ";
          //writing G color
          //offsetting one size of the image
          out_file<<std::to_string(ptr[pix_pos +pic_size_sq])<<" ";
          //writing B color
          //here we offset twice the sice of the image
          out_file<<std::to_string(ptr[pix_pos +pic_size_sq*2])<<"\n";
      }
      //each row needs to be separated by blank line
      out_file<<"\n";
    }
    out_file.close();
    return true;

}
}// end namespace dataset
