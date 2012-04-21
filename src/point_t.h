#ifndef __POINT_T_H
#define __POINT_T_H

#include <stdio.h>
#include "float3.h"

#define SCALE_FACTOR 0.5

struct point_t
{
   int index; // The index of this point.
   float x, y, z; // The coordinates of the point in 3D space.
   int pX, pY; // The screen coordinates of this point.

   point_t() {};

   point_t(int _index, float _x, float _y, float _z) :
      index(_index), x(_x), y(_y), z(_z)
   {
   }

   inline void w2p(int w, int h)
   {
      float dim = SCALE_FACTOR;
      // Convert x.
      float tmpX = x + dim; // Shift.
      pX = (int)(tmpX * (float)(w - 1) / 2 * (1.f / dim)); // Scale.
      // Convert y.
      float tmpY = y + dim; // Shift.
      pY = (int)(tmpY * (float)(h - 1) / 2 * (1.f / dim)); // Scale.
   }

   inline bool isNum(int check)
   {
      return (index == check);
   }

   float3 toF3Screen()
   {
      return float3(pX, pY, 0);
   }

   float3 toF3World()
   {
      return float3(x, y, z);
   }
   
   inline void print(bool world = false)
   {
      //if (!isConverted || world)
      if (world)
      {
         printf("\t%d: %f %f %f\n", index, x, y, z);
      }
      else
      {
         printf("\t%d: %d %d\n", index, pX, pY);
      }
   }
};

#endif
