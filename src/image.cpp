#include "image.h"

using namespace std;

Image::Image(int w, int h, string name):
   width(w), height(h), filename(name)
{
   init();
   //filename = "../output/out.tga";
}

Image::~Image()
{
}

void Image::init()
{
   if (width <= 0 || height <= 0)
   {
      fprintf(stderr, "Invalid Image dimensions.\n");
      exit(EXIT_FAILURE);
   }
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

void Image::print(std::ofstream& out, int r, int g, int b)
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
         //data[j][i].print(myfile);
         int r = (int)(cbuf->data[j][i].v[0] * 255);
         int g = (int)(cbuf->data[j][i].v[1] * 255);
         int b = (int)(cbuf->data[j][i].v[2] * 255);
         //printf("%d %d %d\n", r, g, b);
         //printf("%f %f %f\n", cbuf->data[j][i].v[0], cbuf->data[j][i].v[1], cbuf->data[j][i].v[2]);
         print(myfile, r, g, b);
      }
   }

   cout << "done" << endl;

   myfile.close();
}
