#ifndef _POINT_T_H
#define _POINT_T_H

#include <stdio.h>
#include "vec3.h"

#define SCALE_FACTOR 1.0f

struct point_t
{
   int index; // The index of this point.
   vec_t x, y, z; // The coordinates of the point in 3D space.
   int pX, pY; // The screen coordinates of this point.

   point_t() {};

   point_t(int _index, vec_t _x, vec_t _y, vec_t _z) :
      index(_index), x(_x), y(_y), z(_z)
   {
   }

   inline void w2p(int w, int h, vec_t scale = (vec_t)1.0)
   {
      vec_t dim = scale;
      // Convert x.
      vec_t tmpX = x + dim; // Shift.
      pX = (int)(tmpX * (vec_t)(w - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.
      // Convert y.
      vec_t tmpY = y + dim; // Shift.
      pY = (int)(tmpY * (vec_t)(h - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.
   }

   inline bool isNum(int check)
   {
      return (index == check);
   }

   vec3 toF3Screen()
   {
      return vec3(pX, pY, 0);
   }

   vec3 toF3World()
   {
      return vec3(x, y, z);
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
