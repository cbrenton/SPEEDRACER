/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <string>
#include <iostream>
//#include "pngwriter.h"
#include "vector.h"
#include "colorbuffer.h"

using namespace std;

class Image
{
   public:
      Image(int w, int h, string name);

      ~Image();

      // The width in pixels of the image.
      int width;

      // The height in pixels of the image.
      int height;

      // The name of the file to be output (minus the file extension).
      string filename;

      // Writes a color buffer to the file.
      void write(colorbuffer *buf);

      // Writes a single pixel to the file.
      void writePixel(int x, int y, vec3_t *color);

      // Closes the file.
      void close();

   protected:
      //pngwriter *png;
};
#endif
