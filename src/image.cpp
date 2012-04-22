/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include <iostream>
#include "image.h"

using namespace std;

Image::Image(int w, int h, string name) :
   width(w), height(h)
{
   filename = name;
   cout << "filename: " << filename << endl;
   png = new pngwriter(height, width, 0, filename.c_str());
}

Image::~Image()
{
   delete png;
}

void Image::write(colorbuffer *buf)
{
   for (int x = 0; x < buf->w; x++)
   {
      for (int y = 0; y < buf->h; y++)
      {
         writePixel(x, y, &buf->data[x][y]);
      }
   }
}

void Image::writePixel(int x, int y, float3 *color)
{
   png->plot(x + 1, y + 1, color->x(), color->y(), color->z());
}

void Image::close()
{
   png->close();
}
