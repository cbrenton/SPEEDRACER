#ifndef __Image_H
#define __Image_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "colorbuffer.h"

class Image {
   public:
      Image() {};
      Image(int w, int h, std::string name = "./images/out.tga");
      ~Image();
      void print(std::ofstream& out, float r, float g, float b);
      void writeHeader(std::ofstream& out);
      void write(colorbuffer *cbuf);
   private:
      int width;
      int height;
      std::string filename;
};

#endif
