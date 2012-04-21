/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include <iostream>
#include "image.h"
//#include "Pixel.h"
//#include "Globals.h"

using namespace std;

Image::Image(int w, int h, string name) :
   width(w), height(h)
{
   /*
   // Initialize pixelData.
   pixelData = new Pixel*[width];
   for (int x = 0; x < width; x++)
   {
   pixelData[x] = new Pixel[height];
   for (int y = 0; y < height; y++)
   {
   pixelData[x][y] = Pixel(0.0, 0.0, 0.0);
   }
   }
   */

   filename = name;
   cout << "filename: " << filename << endl;
   png = new pngwriter(height, width, 0, filename.c_str());
}

Image::~Image()
{
   delete png;
   /*
   // Delete pixelData.
   for (int x = 0; x < width; x++)
   {
      delete[] pixelData[width];
   }
   */
}

//void Image::writePixel(int x, int y, const Pixel & pix)
void Image::writePixel(int x, int y, double r, double g, double b)
{
   //png->plot(x, y, pix.c.r, pix.c.g, pix.c.b);
   png->plot(x + 1, y + 1, r, g, b);
}

void Image::close()
{
   png->close();
}
