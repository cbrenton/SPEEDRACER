#ifndef _ZBUFFER_H
#define _ZBUFFER_H

#include <float.h>
#include "vector.h"

struct zbuffer
{
   int w, h;
   vec_t *data;

   zbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new vec_t[w * h];
      for (int i = 0; i < w; i++)
      {
         for (int j = 0; j < h; j++)
         {
            data[i * h + j] = -FLT_MAX;
         }
      }
   }

   ~zbuffer()
   {
      delete [] data;
   }

   void clear()
   {
      for (int i = 0; i < w; i++)
      {
         for (int j = 0; j < h; j++)
         {
            data[i * h + j] = -FLT_MAX;
         }
      }
   }

   vec_t *at(int x, int y)
   {
      if (x < 0 || x > w || y < 0 || y > h)
      {
         printf("Error: index not in range.\n");
         exit(EXIT_FAILURE);
      }
      return &data[x * h + y];
   }

   bool hit(int x, int y, vec_t t)
   {
      if (t <= data[x * h + y])
      {
         return false;
      }
      data[x * h + y] = t;
      return true;
   }
};

#endif
