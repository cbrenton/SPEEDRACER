#ifndef _COLORBUFFER_H
#define _COLORBUFFER_H

#include "float3.h"

struct colorbuffer
{
   int w, h;
   float3 **data;

   colorbuffer(int _w, int _h) :
      w(_w), h(_h)
   {
      data = new float3*[w];
      for (int i = 0; i < w; i++)
      {
         data[i] = new float3[h];
         for (int j = 0; j < h; j++)
         {
            //data[i][j] = Z_INF;
         }
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
