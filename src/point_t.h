#ifndef _POINT_T_H
#define _POINT_T_H

#include <stdio.h>
#include "vector.h"

#define SCALE_FACTOR 1.0f

struct point_t
{
   int index; // The index of this point.
   vec3_t coords; // The coordinates of the point in 3D space.
   int pX, pY; // The screen coordinates of this point.

   point_t()
   {
      index = -1;
      coords.v[0] = 0;
      coords.v[1] = 0;
      coords.v[2] = 0;
      pX = 0;
      pY = 0;
   }

   point_t(int _index, vec_t _x, vec_t _y, vec_t _z) :
      index(_index)
   {
      coords.v[0] = _x;
      coords.v[1] = _y;
      coords.v[2] = _z;
      pX = 0;
      pY = 0;
   }

   point_t(const point_t& pt)
   {
      index = pt.index;
      //*coords = *pt.coords;
      coords[0] = pt.coords[0];
      coords[1] = pt.coords[1];
      coords[2] = pt.coords[2];
      pX = pt.pX;
      pY = pt.pY;
   }

   void w2p(int w, int h, vec_t scale = (vec_t)1.0)
   {
      // Convert x.
      vec_t tmpX = coords.v[0] + scale; // Shift.
      pX = (int)(tmpX * (vec_t)(w - 1) / 2 * ((vec_t)1.0 / scale)); // Scale.
      // Convert y.
      vec_t tmpY = coords.v[1] + scale; // Shift.
      pY = (int)(tmpY * (vec_t)(h - 1) / 2 * ((vec_t)1.0 / scale)); // Scale.
   }

   inline bool isNum(int check)
   {
      return (index == check);
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
         //printf("\t%d: %d %d\n", index, pX, pY);
         printf("\t%d: %d %d\n", index, pX, pY);
      }
   }
};

#endif
