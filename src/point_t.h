#ifndef _POINT_T_H
#define _POINT_T_H

#include <stdio.h>
#include "vec3.h"

#define SCALE_FACTOR 1.0f

struct point_t
{
   int index; // The index of this point.
   vec3 coords; // The coordinates of the point in 3D space.
   int pX, pY; // The screen coordinates of this point.

   point_t() {};

   point_t(int _index, vec_t _x, vec_t _y, vec_t _z) :
      index(_index)
   {
      coords.v[0] = _x;
      coords.v[1] = _y;
      coords.v[2] = _z;
   }

   inline void w2p(int w, int h, vec_t scale = (vec_t)1.0)
   {
      vec_t dim = scale;
      // Convert x.
      vec_t tmpX = coords.v[0] + dim; // Shift.
      pX = (int)(tmpX * (vec_t)(w - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.
      // Convert y.
      vec_t tmpY = coords.v[1] + dim; // Shift.
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
      return coords;
   }
   
   inline void print(bool world = false)
   {
      //if (!isConverted || world)
      if (world)
      {
         printf("\t%d: %f %f %f\n", index, coords.v[0], coords.v[1], coords.v[2]);
      }
      else
      {
         printf("\t%d: %d %d\n", index, pX, pY);
      }
   }
};

#endif
