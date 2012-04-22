#ifndef _ZBUFFER_H
#define _ZBUFFER_H

#include <float.h>

#define Z_INF FLT_MAX

struct zbuffer
{
   int w, h;
   float ***data;

   zbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new float**[w];
      for (int i = 0; i < w; i++)
      {
         data[i] = new float*[h];
         for (int j = 0; j < h; j++)
         {
            data[i][j] = new float;
            *data[i][j] = Z_INF;
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

   bool hit(int x, int y, float t)
   {
      if (*data[x][y] != Z_INF && t >= *data[x][y])
      {
         return false;
      }
      *data[x][y] = t;
      return true;
   }
};

#endif
