/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <string>
#include "pngwriter.h"
#include "float3.h"

using namespace std;

//class Pixel;

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

      // The pixel data currently stored in the image.
      //Pixel **pixelData;

      // Writes a single pixel to the file.
      //void writePixel(int x, int y, const Pixel & pix) {};
      //void writePixel(int x, int y, double r, double g, double b);
      void writePixel(int x, int y, float3 *color);

      // Closes the file.
      void close();

   protected:
      pngwriter *png;
};
#endif
