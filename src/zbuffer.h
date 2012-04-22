#ifndef _ZBUFFER_H
#define _ZBUFFER_H

#include <float.h>
#include "vec3.h"

struct zbuffer
{
   int w, h;
   vec_t ***data;

   zbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new vec_t**[w];
      for (int i = 0; i < w; i++)
      {
         data[i] = new vec_t*[h];
         for (int j = 0; j < h; j++)
         {
            data[i][j] = new vec_t;
            *data[i][j] = FLT_MAX;
         }
      }
   }

   ~zbuffer()
   {
      for (int i = 0; i < w; i++)
      {
         for (int j = 0; j < h; j++)
         {
            delete data[i][j];
         }
         delete [] data[i];
      }
      delete [] data;
   }

   bool hit(int x, int y, vec_t t)
   {
      if (t >= *data[x][y])
      {
         return false;
      }
      *data[x][y] = t;
      return true;
   }
};

#endif
