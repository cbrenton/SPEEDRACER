#include "image.h"

using namespace std;

Image::Image(int w, int h, string name):
   width(w), height(h), filename(name)
{
   if (width <= 0 || height <= 0)
   {
      fprintf(stderr, "Invalid Image dimensions.\n");
      exit(EXIT_FAILURE);
   }
}

Image::~Image()
{
}

void Image::writeHeader(ofstream& out)
{
   out << '\0'
      << '\0'
      << '\2' // Uncompressed RGB
      << '\0' << '\0'
      << '\0' << '\0'
      << '\0'
      << '\0' << '\0' // X origin
      << '\0' << '\0' // Y origin
      << (char) (width & 0xFF)
      << (char) ((width >> 8) & 0xFF)
      << (char) (height & 0xFF)
      << (char) ((height >> 8) & 0xFF)
      << (char) 32 // 32 bit bitmap
      << '\0';
}

void Image::print(std::ofstream& out, float r, float g, float b)
{
   out << (char)(b * 255)
      << (char)(g * 255)
      << (char)(r * 255)
      << (char)(255);
}

void Image::write(colorbuffer *cbuf)
{
   char *name = (char*)filename.c_str();
   ofstream myfile;
   myfile.open(name);
   if (!myfile)
   {
      cerr << "Error: unable to open " << name << endl;
      exit(EXIT_FAILURE);
   }
   else
   {
      cout << "Writing to file " << name << "...";
   }

   writeHeader(myfile);

   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j < width; j++)
      {
         float r = cbuf->data[j][i].v[0];
         float g = cbuf->data[j][i].v[1];
         float b = cbuf->data[j][i].v[2];
         print(myfile, r, g, b);
      }
   }

   cout << "done" << endl;

   myfile.close();
}
