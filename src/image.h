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
      Image(int w, int h, std::string name);
      ~Image();
      void setSize(int w, int h);
      void init();
      void print(std::ofstream& out, int r, int g, int b);
      void writeHeader(std::ofstream& out);
      void write(colorbuffer *cbuf);
      int width;
      int height;
      std::string filename;
};

#endif
