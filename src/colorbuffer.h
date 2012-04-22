#ifndef _COLORBUFFER_H
#define _COLORBUFFER_H

#include "vec3.h"

struct colorbuffer
{
   int w, h;
   vec3 **data;

   colorbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new vec3*[w];
      for (int i = 0; i < w; i++)
      {
         data[i] = new vec3[h];
      }
   }

   ~colorbuffer()
   {
      for (int i = 0; i < w; i++)
      {
         delete [] data[i];
      }
      delete [] data;
   }
};

#endif
