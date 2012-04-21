#ifndef __POINT_T_H
#define __POINT_T_H

#include <stdio.h>
#include "float3.h"

struct point_t
{
   int index; // The index of this point.
   float x, y, z; // The coordinates of the point in 3D space.
   int pX, pY; // The screen coordinates of this point.
   bool isConverted;

   point_t() {};

   point_t(int _index, float _x, float _y, float _z) :
      index(_index), x(_x), y(_y), z(_z)
   {
      isConverted = false;
   }

   inline void w2p(int w, int h)
   {
      // Convert x.
      float tmpX = x + 1.f; // Shift.
      pX = (int)(tmpX * (float)(w - 1) / 2); // Scale.
      // Convert y.
      float tmpY = y + 1.f; // Shift.
      pY = (int)(tmpY * (float)(h - 1) / 2); // Scale.
 
      isConverted = true;
   }

   inline bool isNum(int check)
   {
      return (index == check);
   }

   float3 toF3()
   {
      //return float3(x, y, z);
      return float3(pX, pY, 0);
   }
   
   inline void print(bool world = false)
   {
      if (!isConverted || world)
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
