#include <cassert>
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

//coursera

bool load_coursera_cat(const std::string &outpath, Matrix<uint8_t> &X,
                       Matrix<uint8_t> &Y, std::vector<uint8_t> &Xstorage,
                       std::vector<uint8_t> &Ystorage, bool add_bias, bool load_validation) {
  const uint32_t IMAGE_DATA_SIZE = 64u * 64u * 3;
  uint32_t IMAGES_PER_FILE = 209u;
  if (load_validation) {
    IMAGES_PER_FILE = 50;
  }

  Xstorage.reserve(IMAGE_DATA_SIZE * IMAGES_PER_FILE +
                   IMAGES_PER_FILE * static_cast<uint32_t>(add_bias));
  Ystorage.reserve( IMAGES_PER_FILE);

  //const std::string Xpath{outpath + "train_X_209_12288_v2.txt"};
  std::string Xpath{outpath + "train_X_209_12288.txt"};
  std::string Ypath{outpath + "train_Y_209_1.txt"};
  if (load_validation) {
    Xpath = outpath + "test_X_50_12288.txt";
    Ypath = outpath + "test_Y_50_1.txt";
  }

  //lets read X
  
  std::ifstream trainX(Xpath);
  if (!trainX) {
    return false;
  }

  //loading X
  std::stringstream buffer;
  buffer << trainX.rdbuf();
  trainX.close();
  std::string number_as_string;
  std::string line;
  if (!add_bias) {
    while (std::getline(buffer, line)) {
      std::istringstream ss(line);
      while (std::getline(ss, number_as_string, ' ')) {
        Xstorage.push_back(std::stof(number_as_string));
      }
    }
  }
  else
  {
    while (std::getline(buffer, line)) {
      std::istringstream ss(line);
      //adding 1 for bias, since the data will be normalized we 
      //will add 255 which normalized is 1
      Xstorage.push_back(255); 
      while (std::getline(ss, number_as_string, ' ')) {
        Xstorage.push_back(std::stof(number_as_string));
      }
    }
  
  }
  std::ifstream trainY(Ypath);
  if (!trainY) {
    return false;
  }

  std::stringstream bufferY;
  bufferY << trainY.rdbuf();
  trainY.close();
  while (std::getline(bufferY, number_as_string,',')) {
    Ystorage.push_back(std::stof(number_as_string));
  }
  X.data = Xstorage.data();
  X.size_x = IMAGES_PER_FILE;
  X.size_y = add_bias ? IMAGE_DATA_SIZE+1 : IMAGE_DATA_SIZE;
  Y.data = Ystorage.data();
  Y.size_x = 1;
  Y.size_y = IMAGES_PER_FILE;
  return true;
}


bool dump_image_from_coursera_cat_dataset(const std::string &outpath,
                                      Matrix<uint8_t> &data, uint32_t index)
{
    
    std::ofstream out_file;
    out_file.open(outpath, std::ios::out);
    if (!out_file) {
      return false;
    }
    //
    // size of the picture is the colum count of the matrix, since each row is a
    // full
    // picture, the data is first all the r, then all the g then all the b, so
    // if
    // we divide by 3 we get the total number of pixels, taking squre root since
    // image from cifar 10 is squared
    uint32_t pic_size = sqrt(std::floor((data.size_y/3.0))); 
    //shifting the pointer of the image by how many images we wish to skip
    const uint8_t* ptr= data.data + data.size_y*index ;

    for (uint32_t r = 0; r < pic_size; ++r) {
      for (uint32_t c = 0; c < pic_size; ++c) {
          //adding pixel coordinate
          //coordinate of the pixel, needed to do some permutation in order 
          //for the picture to show oriented correctly
          out_file<<std::to_string(c)<<" "<<std::to_string(64-r)<<" ";
          //writing R color
          uint32_t pix_pos = r*(pic_size*3)+c*3;
          out_file<<std::to_string(int(ptr[pix_pos]))<<" ";
          //writing G color
          //offsetting one size of the image
          out_file<<std::to_string(int(ptr[pix_pos +1]))<<" ";
          //writing B color
          //here we offset twice the sice of the image
          out_file<<std::to_string(int(ptr[pix_pos +2]))<<"\n";
      }
      //each row needs to be separated by blank line
      out_file<<"\n";
    }
    out_file.close();
    return true;

}

void normalize_image_dataset(Matrix<uint8_t> &data, Matrix<float> &dataout,
                             float norm_value) {
  assert(data.size_x == dataout.size_x);
  assert(data.size_y == dataout.size_y);
  float inv_norm = 1.0f / norm_value;
  uint32_t total_size = data.total_size();

  const uint8_t *const inp = data.data;
  float *const op = dataout.data;

  for (uint32_t i = 0; i < total_size; ++i) {
    op[i] = inv_norm * static_cast<float>(inp[i]);
  }
}

}// end namespace dataset
