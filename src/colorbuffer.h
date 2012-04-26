#ifndef _COLORBUFFER_H
#define _COLORBUFFER_H

#include "vector.h"

struct colorbuffer
{
   int w, h;
   vec3_t *data;

   colorbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new vec3_t[w * h];
   }

   ~colorbuffer()
   {
      delete [] data;
   }

   vec3_t *at(int x, int y)
   {
      if (x < 0 || x > w || y < 0 || y > h)
      {
         printf("Error: index not in range.\n");
         exit(EXIT_FAILURE);
      }
      return &data[x * h + y];
   }
};

#endif
