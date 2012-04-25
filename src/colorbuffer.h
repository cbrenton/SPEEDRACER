#ifndef _COLORBUFFER_H
#define _COLORBUFFER_H

#include "vector.h"

struct colorbuffer
{
   int w, h;
   vec3_t **data;

   colorbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new vec3_t*[w];
      for (int i = 0; i < w; i++)
      {
         data[i] = new vec3_t[h];
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
